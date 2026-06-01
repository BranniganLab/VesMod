#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:04:46 2025.

@author: js2746
"""
from pathlib import Path
import json
import numpy as np
from vesmod.EdgeMod import read_and_format_csv, calc_sq_amplitudes, interpolate_indices_vectorized, filter_data, fit_spectrum_to_theory_lmfit
from collections import namedtuple
from statsmodels.tsa import stattools
from lmfit.models import ExponentialModel
FrameCount = namedtuple("FrameCount", ['total_frames', 'useable_frames', 'pct_useable'])
MiniSpectrum = namedtuple("MiniSpectrum", ['modes', 'avg_amps2', 'std_amps2'])


class SingleSpectrum:
    """
    Contains the average squared amplitudes from one vesicle \
    video only. Multiple SingleSpectrum objects can be combined into a \
    CombinedSpectra object.

    Attributes
    ----------
    path : Path
        The path to the csv file that holds the edge extraction data pertaining\
        to this spectrum.
    unfiltered_frames : ndarray
        2D array containing the vesicle radius for each theta bin for each frame,\
        before filter is applied.
    modes : ndarray of ints or None
        The modes for each amplitude. Each value is an integer. If \
        useable_frames == 0, this is set to None.
    avg_amps2 : ndarray of floats or None
        The squared amplitudes of each mode, averaged over the trajectory. If \
        useable_frames == 0, this is set to None.
    r0 : float
        The average vesicle radius, in arbitrary units.

    """

    def __init__(self, path, Ntheta=None, frame_cutoff=None, filter_type='strict'):
        """
        Create a SingleSpectrum object.

        Parameters
        ----------
        path : str or Path
            The path to the file you want to analyze.
        Ntheta : int or None, optional
            The number of theta values to store.
        frame_cutoff : int or None, optional
            The number of frames to retain in your trajectory. The default is None.

        """
        # make sure path is correct
        assert isinstance(path, (str, Path)), "path must be a str or a pathlib Path object."
        if isinstance(path, str):
            path = Path(path)
        assert path.is_file() is True, "path does not appear to point to a file."
        ftype = path.suffix
        assert ftype in ['.csv', '.npy'], "Edge extraction outputs should be .csv or .npy files."

        # make sure frame_cutoff is either None or is an int
        if frame_cutoff is not None:
            assert isinstance(frame_cutoff, int), "frame_cutoff must either be None or an int."

        if Ntheta is not None:
            assert isinstance(Ntheta, int), "Ntheta must either be None or an int"

        # read in the file specified by path
        if ftype == '.csv':
            input_data, _ = read_and_format_csv(path)
        elif ftype == ".npy":
            input_data = np.load(path)

        # prune the trajectory if frame_cutoff specified
        if frame_cutoff is not None and frame_cutoff < input_data.shape[0]:
            input_data = input_data[:frame_cutoff, :]

        # ensure that dtheta is equal for each sample
        if Ntheta is not None and Ntheta < input_data.shape[1]:
            zero_to_ntheta = np.linspace(0, Ntheta - 1, Ntheta)
            new_evenly_spaced_indices = zero_to_ntheta * (input_data.shape[1] / Ntheta)
            input_data = interpolate_indices_vectorized(input_data, new_evenly_spaced_indices)
        elif Ntheta is not None and Ntheta > input_data.shape[1]:
            raise IndexError(f"Input array has {input_data.shape[1]} columns; cannot interpolate into {Ntheta} columns")

        self.unfiltered_frames = input_data
        self._total_frames = self.unfiltered_frames.shape[0]

        # remove frames that contain bad edge extraction results
        filtered_useable_data, filtered_full_data = filter_data(self.unfiltered_frames, filter_type)

        self._useable_frames = filtered_useable_data.shape[0]

        if self._useable_frames == 0:
            # Trajectory is completely unuseable (all frames were removed by filtering step)
            self.modes = None
            self.avg_amps2 = None
            self.path = path
            self.r0 = None
            self._filtered_spectra = None
            self.kC_3_8 = None
            self.kC_8_13 = None
        else:
            self.r0 = np.mean(filtered_useable_data)
            N_samples = filtered_useable_data.shape[1]
            norm = 1. / (self.r0 * N_samples)
            amps2, self.modes = calc_sq_amplitudes(filtered_useable_data, norm)
            self.avg_amps2 = np.mean(amps2.real, axis=0)
            self.path = path
            self._filtered_spectra, _ = calc_sq_amplitudes(filtered_full_data, norm)
            self.kC_3_8 = fit_spectrum_to_theory_lmfit(self.isolate_mode_range(3, 8), 500, free_sigma=True)
            self.kC_8_13 = fit_spectrum_to_theory_lmfit(self.isolate_mode_range(8, 13), 500, free_sigma=True)

        self.to_json(path.with_suffix(".json"))

    @property
    def ideal_block_size(self):
        """
        Determine the block size that will eliminate all NaNs once block averaging\
        is performed. This is defined as the longest contiguous block of NaNs in\
        the array column.

        Returns
        -------
        int.

        """
        temp_block_size = 0
        size_list = []
        for val in np.isnan(self._filtered_spectra[:, 0]):
            if val:
                temp_block_size += 1
            else:
                size_list.append(temp_block_size)
                temp_block_size = 0
        size_list.append(temp_block_size)
        final_block_size = max(size_list) + 1
        return int(final_block_size)

    @property
    def frame_count(self):
        """
        Count the total number of frames in this SingleSpectrum, count how \
        many frames are useable (successful edge detection and passed \
        filtering), and determine the percentage of useable frames.

        Returns
        -------
        FrameCount : namedtuple
            total_frames : int
            useable_frames : int
            pct_useable : float

        """
        pct_useable = self._useable_frames / self._total_frames
        return FrameCount(self._total_frames, self._useable_frames, pct_useable)

    def block_average(self, block_size=None):
        """
        Calculate block average of self._filtered_spectra using specified step\
        size.

        Parameters
        ----------
        block_size : int
            The step size to use when block averaging.

        Returns
        -------
        block_avg : ndarray
            2D ndarray of block averaged values.

        """
        if not block_size:
            block_size = self.ideal_block_size
        assert isinstance(block_size, (int, np.int64)), "block_size must be an integer"
        assert block_size > 0, "block_size must be positive"
        num_rows = int(self._filtered_spectra.shape[0] // block_size)
        block_avg = np.zeros((num_rows, self._filtered_spectra.shape[1]))
        for row in range(num_rows):
            block_avg[row, :] = np.nanmean(self._filtered_spectra[(block_size * row):(block_size * (row + 1)), :].real, axis=0)
        return block_avg

    def isolate_mode_range(self, lower_bound, upper_bound, filtered_full=False):
        """
        Return all modes greater than or equal to lower_bound and less than \
        upper_bound and their associated avg squared amplitudes.

        Returns
        -------
        MiniSpectrum : namedtuple
            modes : ndarray
                The modes within range [lower_bound:upper_bound).
            avg_amps2 : ndarray
                The avg squared amplitudes of those modes.
            std_amps2 : ndarray
                The standard deviation of the avg_amps2

        """
        if self.modes is not None:
            mask1 = self.modes >= lower_bound
            mask2 = self.modes < upper_bound
            combined_mask = mask1 & mask2
            if filtered_full:
                return MiniSpectrum(self.modes[combined_mask], self._filtered_spectra[:, combined_mask].real, None)
            else:
                return MiniSpectrum(self.modes[combined_mask], self.avg_amps2[combined_mask], None)
        else:
            return None

    def measure_autocorrelation(self, block_size, nlags):
        """
        Measure the autocorrelation of the squared amplitudes corresponding to \
        each non-negative mode. If there are too few samples to calculate auto-\
        correlation with nlags lags, return "NULL".

        Parameters
        ----------
        block_size : int
            The block size to use for block averaging.
        nlags : int
            The number of lags (taus) to calculate autocorrelation for.

        Returns
        -------
        autocorrelations : ndarray
            2D array containing autocorrelation coefficients for each lag for \
            each mode. (dimensions = nmodes X nlags + 1)

        """
        positive_modes = self.isolate_mode_range(0, np.inf)
        num_rows = int(self._filtered_spectra.shape[0] // block_size)
        if num_rows < nlags + 1:
            return "NULL", positive_modes.modes
        nmodes = len(positive_modes.modes)
        block_avg = self.block_average(block_size)
        autocorrelations = np.zeros((nmodes, nlags + 1))
        for mode in range(nmodes):
            autocorrelations[mode, :] = stattools.acf(block_avg[:, mode], nlags=nlags)
        return autocorrelations, positive_modes

    def measure_characteristic_decay_lengths(self, block_size, nlags):
        """
        Measure the characteristic decay length (tau) of the autocorrelation \
        of each non-negative mode in the spectrum. If there are too few samples\
        to make the calculation, return "NULL".

        Parameters
        ----------
        block_size : int
            The block size to use for block averaging.
        nlags : int
            The number of lags to use in the autocorrelation computation.

        Returns
        -------
        ndarray
            1D array of decay lengths, corresponding to each non-negative mode\
            in the spectrum.
        ndarray
            1D array of the x axis used (can be used for plotting).

        """
        decay_lengths = []
        autocorrelations, modes = self.measure_autocorrelation(block_size, nlags)
        if isinstance(autocorrelations, str):
            return "NULL", "NULL", modes
        x_axis = np.linspace(0, nlags * block_size, nlags + 1)
        for acf in autocorrelations:
            model = ExponentialModel(nan_policy='propagate')
            params = model.make_params(amplitude={'value': 1, 'vary': False}, decay=.7 * block_size)
            out = model.fit(acf, params, x=x_axis)
            decay_lengths.append(out.best_values['decay'])
        assert len(decay_lengths) == autocorrelations.shape[0], "something went wrong"
        return np.array(decay_lengths), x_axis, modes

    def calc_correlation_time(self, dx):
        pass

    def calc_n_decorr_samples(self, block_size, nlags, bounded=False):
        """
        Calculate the number of decorrelated samples for each mode. Number of\
        decorrelated samples is defined as:\
        decay length / (number of frames // block size).

        Parameters
        ----------
        block_size : int
            The block size to use for block averaging.
        nlags : int
            The number of lags to use in the autocorrelation computation.
        bounded : boolean
            If False, number of decorrelated samples can be less than one. If \
            True, number of decorrelated samples must be 1 or greater.

        Returns
        -------
        ndarray
            1D array of number of decorrelated samples present in the replica.

        """
        decay_lengths, _, modes = self.measure_characteristic_decay_lengths(block_size, nlags)
        if isinstance(decay_lengths, str):
            return np.ones(modes.shape[0])
        if bounded:
            decay_lengths[decay_lengths < 1] = 1
        n_samples = self.total_frames // block_size
        return n_samples / decay_lengths

    def _to_dict(self, include_arrays=True):
        """Convert class attributes to a dict."""
        data = {
            "path": str(self.path) if getattr(self, "path", None) is not None else None,
            "r0": float(self.r0) if getattr(self, "r0", None) is not None else None,
            "frame_count": self.frame_count,
            "kC_3_8": self.kC_3_8,
            "kC_8_13": self.kC_8_13,
        }

        if include_arrays:
            data["modes"] = (
                self.modes.tolist() if getattr(self, "modes", None) is not None else None
            )
            data["avg_amps2"] = (
                self.avg_amps2.tolist() if getattr(self, "avg_amps2", None) is not None else None
            )

        return data

    def to_json(self, outfile, include_arrays=True, indent=2):
        """Save class attributes to json."""
        outfile = Path(outfile)
        with outfile.open("w", encoding="utf-8") as f:
            json.dump(self._to_dict(include_arrays=include_arrays), f, indent=indent)
