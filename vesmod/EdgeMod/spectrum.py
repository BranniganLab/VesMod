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
from vesmod.VesEdge import VesicleVideo

MiniSpectrum = namedtuple("MiniSpectrum", ['modes', 'avg_amps2', 'std_amps2'])


class Spectrum:
    """
    Calculate the fluctuation spectrum of a vesicle video.

    Attributes
    ----------
    modes : ndarray[int]
        The modes for each amplitude. Each value is an integer.
    avg_amps2 : ndarray[float]
        The squared amplitudes of each mode, averaged over the trajectory.
    r0 : float
        The average vesicle radius, in microns.

    """

    def __init__(
        self,
        edges_over_time: str | Path | VesicleVideo,
        Ntheta=None,
        frame_cutoff=None
    ) -> None:
        """
        Create a SingleSpectrum object.

        Parameters
        ----------
        edges_over_time : str or Path or VesicleVideo
            The path (str or Path) to the .npy file containing edge extraction r_vals or
            the VesicleVideo object itself.
        Ntheta : int or None, optional
            The number of theta values to store. Default is None.
        frame_cutoff : int or None, optional
            The number of frames to retain in your trajectory. Default is None.

        """
        if isinstance(edges_over_time, VesicleVideo):
            input_data = edges_over_time.r_vals
        elif isinstance(edges_over_time, (Path, str)):
            if isinstance(edges_over_time, str):
                edges_over_time = Path(edges_over_time)
            if not edges_over_time.is_file():
                raise ValueError("edges_over_time does not appear to be a file.")
            if edges_over_time.suffix != '.npy':
                raise ValueError("edges_over_time must end in .npy")
            input_data = np.load(edges_over_time)
        else:
            raise TypeError("edges_over_time must be a str, pathlib Path, or VesicleVideo.")

        # make sure frame_cutoff and Ntheta are either None or a positive int
        for var, varname in zip([frame_cutoff, Ntheta], ["frame_cutoff", "Ntheta"]):
            if not isinstance(var, (int, NoneType)):
                raise TypeError(f"{varname} must either be None or an int.")
            if (isinstance(var, int)) and (var <= 0):
                raise ValueError(f"{varname} must be a positive int.")

        # prune the trajectory if frame_cutoff specified
        if frame_cutoff is not None and frame_cutoff < input_data.shape[0]:
            input_data = input_data[:frame_cutoff, :]

        # downsample to Ntheta to ensure that dtheta is equal across all replicas
        if Ntheta is not None and Ntheta < input_data.shape[1]:
            zero_to_ntheta = np.linspace(0, Ntheta - 1, Ntheta)
            new_evenly_spaced_indices = zero_to_ntheta * (input_data.shape[1] / Ntheta)
            input_data = downsample_to_new_indices(input_data, new_evenly_spaced_indices)
        elif Ntheta is not None and Ntheta > input_data.shape[1]:
            raise IndexError(f"Input array has {input_data.shape[1]} columns; cannot downsample into {Ntheta} columns")

        self.r0 = np.mean(input_data)
        self.avg_amps2 = self._calc_avg_sq_amplitudes(input_data)
        self.modes = self._calc_integer_modes()
        self.kC = None
        self.surface_tension = None

    def _calc_avg_sq_amplitudes(self, r_vals_over_time: np.ndarray) -> np.ndarray:
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

    def _calc_integer_modes(self) -> np.ndarray[int]:
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

    def isolate_mode_range(self, lower_bound: int, upper_bound: int) -> MiniSpectrum:
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
        lower_bound: int = 3,
        upper_bound: int = 8,
        lmax: int = 500,
        free_sigma: bool = True
    ) -> tuple(float, float):
        """
        Fit specific range of self.avg_amps2 to theoretical prediction.

        Parameters
        ----------
        lower_bound, upper_bound : ints
            Fit modes greater than or equal to lower_bound and less than upper_bound.
        lmax : int
            Maximum iteration index in theoretical summation. Default is 500.
        free_sigma : bool
            If True, allow surface tension (sigma) to vary. If False, set sigma
            to zero and do not let it vary during fitting.

        Returns
        -------
        tuple[float, float]
            The best fitting kC and sigma values within the fitting range.

        Side Effects
        ------------
        Saves kC to self.kC and sigma to self.surface_tension.

        """
        fitting_range = self.isolate_mode_range(lower_bound, upper_bound)
        fit = fit_spectrum_to_theory_lmfit(fitting_range, lmax, free_sigma)
        self.kC, self.surface_tension = fit
        return fit

    def _to_dict(self, include_arrays=True) -> dict:
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
            "r0": float(self.r0) if getattr(self, "r0", None) is not None else None,
            "kC": float(self.kC) if getattr(self, "kC", None) is not None else None,
            "surface_tension": float(self.surface_tension) if getattr(self, "surface_tension", None) is not None else None,
        }

        if include_arrays:
            data["modes"] = (
                self.modes.tolist() if getattr(self, "modes", None) is not None else None
            )
            data["avg_amps2"] = (
                self.avg_amps2.tolist() if getattr(self, "avg_amps2", None) is not None else None
            )

        return data

    def to_json(
        self,
        outfile: str | Path,
        include_arrays: bool = True,
        indent: int = 2,
    ) -> None:
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
