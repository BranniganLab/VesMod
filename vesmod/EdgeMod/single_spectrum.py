#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:04:46 2025.

@author: js2746
"""
from pathlib import Path
from collections import namedtuple
import json
import numpy as np
from .spectrum_utils import read_and_format_csv, calc_sq_amplitudes, interpolate_indices_vectorized, fit_spectrum_to_theory_lmfit

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

        self.r0 = np.mean(input_data)
        N_samples = input_data.shape[1]
        norm = 1. / (self.r0 * N_samples)
        amps2, self.modes = calc_sq_amplitudes(input_data, norm)
        self.avg_amps2 = np.mean(amps2.real, axis=0)
        self.path = path
        self.kC = fit_spectrum_to_theory_lmfit(self.isolate_mode_range(3, 8), 500, free_sigma=True)

        self.to_json(path.with_suffix(".json"))

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

    def _to_dict(self, include_arrays=True):
        """Convert class attributes to a dict."""
        data = {
            "path": str(self.path) if getattr(self, "path", None) is not None else None,
            "r0": float(self.r0) if getattr(self, "r0", None) is not None else None,
            "kC": self.kC,
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
