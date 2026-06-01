#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:04:46 2025.

@author: js2746
"""
from pathlib import Path
from types import NoneType
from collections import namedtuple
import json
import numpy as np
from .spectrum_utils import downsample_to_new_indices, fit_spectrum_to_theory_lmfit

MiniSpectrum = namedtuple("MiniSpectrum", ['modes', 'avg_amps2', 'std_amps2'])


class SingleSpectrum:
    """
    Calculate the fluctuation spectrum of a vesicle video.

    Attributes
    ----------
    path : Path
        The path to the npy file that holds the edge extraction data pertaining\
        to this spectrum.
    modes : ndarray of ints or None
        The modes for each amplitude. Each value is an integer. If \
        useable_frames == 0, this is set to None.
    avg_amps2 : ndarray of floats or None
        The squared amplitudes of each mode, averaged over the trajectory. If \
        useable_frames == 0, this is set to None.
    r0 : float
        The average vesicle radius, in arbitrary units.

    """

    def __init__(self, path, Ntheta=None, frame_cutoff=None):
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
        if not isinstance(path, (str, Path)):
            raise TypeError("Path must be a str or a pathlib Path object.")
        if isinstance(path, str):
            path = Path(path)
        if not path.is_file():
            raise ValueError("path does not appear to point to a file.")
        if path.suffix != '.npy':
            raise ValueError("path must end in .npy")

        # make sure frame_cutoff and Ntheta are either None or a positive int
        for var, varname in zip([frame_cutoff, Ntheta], ["frame_cutoff", "Ntheta"]):
            if not isinstance(var, (int, NoneType)):
                raise TypeError(f"{varname} must either be None or an int.")
            if (isinstance(var, int)) and (var <= 0):
                raise ValueError(f"{varname} must be a positive int.")

        # read in the file specified by path
        input_data = np.load(path)

        # prune the trajectory if frame_cutoff specified
        if frame_cutoff is not None and frame_cutoff < input_data.shape[0]:
            input_data = input_data[:frame_cutoff, :]

        # downsample to Ntheta to ensure that dtheta is equal across all replicas
        if Ntheta is not None and Ntheta < input_data.shape[1]:
            zero_to_ntheta = np.linspace(0, Ntheta - 1, Ntheta)
            new_evenly_spaced_indices = zero_to_ntheta * (input_data.shape[1] / Ntheta)
            downsampled_data = downsample_to_new_indices(input_data, new_evenly_spaced_indices)
        elif Ntheta is not None and Ntheta > input_data.shape[1]:
            raise IndexError(f"Input array has {input_data.shape[1]} columns; cannot downsample into {Ntheta} columns")

        self.r0 = np.mean(downsampled_data)
        self.avg_amps2 = self.calc_avg_sq_amplitudes(downsampled_data)
        self.modes = self.calc_integer_modes(downsampled_data)

    def calc_avg_sq_amplitudes(self, r_vals_over_time: np.ndarray) -> np.ndarray:
        """
        Calculate the normalized Fourier transform, then square and average.

        Parameters
        ----------
        r_vals_over_time : np.ndarray
            2D array where each row contains the r values for a given frame of
            a vesicle video.
        """
        # Calculate normalizing factor
        n_samples = r_vals_over_time.shape[1]
        norm = 1. / (self.r0 * n_samples)

        # Fourier transform and normalize
        amps = np.fft.fft(r_vals_over_time, axis=1, norm='backward') * norm

        # Multiply by complex conjugate
        amps2 = amps * amps.conj()

        # Take average over time
        avg_amps2 = np.mean(amps2.real, axis=0)
        return avg_amps2

    def calc_integer_modes(self) -> np.ndarray[int]:
        """
        Calculate the integer Fourier modes q for your spectrum.

        numpy's fftfreq will return normalized floats. Multiply by # of modes
        (positive and negative) to get ints.

        Returns
        -------
        np.ndarray[int]
        """
        freqs = np.fft.fftfreq(self.avg_amps2.shape[1])
        modes = np.round(freqs * self.avg_amps2.shape[1]).astype(int)
        return modes

    def isolate_mode_range(self, lower_bound, upper_bound, filtered_full=False):
        """
        Return all modes greater than or equal to lower_bound and less than \
        upper_bound, and their associated avg squared amplitudes.

        Returns
        -------
        MiniSpectrum : namedtuple

        """
        if self.modes is None:
            raise AttributeError("There are no modes; Cannot return mode range.")
        mask1 = self.modes >= lower_bound
        mask2 = self.modes < upper_bound
        combined_mask = mask1 & mask2
        return MiniSpectrum(self.modes[combined_mask], self.avg_amps2[combined_mask], None)

    def extract_kc_from_fit(
        self,
        mode_low: int = 3,
        mode_high: int = 8,
        lmax: int = 500,
        free_sigma: bool = True
    ) -> float:
        """
        Fit specific range of self.avg_amps2 to theoretical prediction.

        Parameters
        ----------
        mode_low, mode_high : ints
            Fit modes greater than or equal to mode_low and less than mode_high.
        lmax : int
            Maximum iteration index in theoretical summation. Default is 500.
        free_sigma : bool
            If True, allow surface tension (sigma) to vary. If False, set sigma
            to zero and do not let it vary during fitting.

        Returns
        -------
        float
            The fit kC from the portion of the spectrum defined by fitting_range.

        Side Effects
        ------------
        Saves kC to self.kC.
        """
        fitting_range = self.isolate_mode_range(mode_low, mode_high)
        kC = fit_spectrum_to_theory_lmfit(fitting_range, lmax, free_sigma)
        self.kC = kC
        return kC

    def _to_dict(self, include_arrays=True):
        """
        Convert class attributes to a dict.

        Parameters
        ----------
        include_arrays : bool, optional
            If True, include modes and avg_amps2 values. Default is True.

        Returns
        -------
        dict
        """
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
        """
        Save class attributes to json.

        Parameters
        ----------
        include_arrays : bool, optional
            If True, include modes and avg_amps2 values. Default is True.

        Side Effects
        ------------
        Saves json to file system.
        """
        outfile = Path(outfile).with_suffix('.json')
        with outfile.open("w", encoding="utf-8") as f:
            json.dump(self._to_dict(include_arrays=include_arrays), f, indent=indent)
