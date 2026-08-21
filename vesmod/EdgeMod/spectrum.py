#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:04:46 2025.

@author: js2746
"""
from pathlib import Path
from types import NoneType
import json
import numpy as np
from vesmod.VesEdge import VesicleEdges
from .spectrum_utils import fit_spectrum_to_theory_lmfit, calc_tension_from_reduced_tension, MiniSpectrum


class Spectrum:
    """
    Calculate the fluctuation spectrum of a vesicle edge trajectory.

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
        edges_over_time: str | Path | VesicleEdges,
        frame_cutoff=None
    ) -> None:
        """
        Create a SingleSpectrum object.

        Parameters
        ----------
        edges_over_time : str or Path or VesicleEdges
            Path to a ``.npy`` file containing accepted edge radii, or a
            ``VesicleEdges`` object on which QC has already been run. When a
            ``VesicleEdges`` object is supplied, only accepted detections are
            included.
        frame_cutoff : int or None, optional
            The number of frames to retain in your trajectory. Default is None.

        """
        if isinstance(edges_over_time, VesicleEdges):
            accepted_radii = [
                detection.radii_microns
                for detection in edges_over_time.accepted_detections
            ]
            if not accepted_radii:
                raise ValueError(
                    "VesicleEdges contains no accepted edge detections."
                )
            lengths = {
                radii.shape[0]
                for radii in accepted_radii
            }
            if len(lengths) > 1:
                raise ValueError(
                    "Accepted VesicleEdges detections have inconsistent "
                    "numbers of angular samples."
                )
            input_data = np.stack(accepted_radii)
        elif isinstance(edges_over_time, (Path, str)):
            if isinstance(edges_over_time, str):
                edges_over_time = Path(edges_over_time)
            if not edges_over_time.is_file():
                raise ValueError("edges_over_time does not appear to be a file.")
            if edges_over_time.suffix != '.npy':
                raise ValueError("edges_over_time must end in .npy")
            input_data = np.load(edges_over_time)
        else:
            raise TypeError(
                "edges_over_time must be a str, pathlib Path, or VesicleEdges."
            )

        if not isinstance(frame_cutoff, (int, NoneType)):
            raise TypeError("frame_cutoff must either be None or an int.")
        if isinstance(frame_cutoff, int) and frame_cutoff <= 0:
            raise ValueError("frame_cutoff must be a positive int.")

        if frame_cutoff is not None and frame_cutoff < input_data.shape[0]:
            input_data = input_data[:frame_cutoff, :]

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
        n_samples = r_vals_over_time.shape[1]
        norm = 1. / (self.r0 * n_samples)
        amps = np.fft.fft(r_vals_over_time, axis=1, norm='backward') * norm
        amps2 = amps * amps.conj()
        avg_amps2 = np.mean(amps2.real, axis=0)
        return avg_amps2

    def _calc_integer_modes(self) -> np.ndarray[int]:
        """Calculate integer Fourier modes q for the spectrum."""
        freqs = np.fft.fftfreq(self.avg_amps2.shape[0])
        modes = np.round(freqs * self.avg_amps2.shape[0]).astype(int)
        return modes

    def isolate_mode_range(self, lower_bound: int, upper_bound: int) -> MiniSpectrum:
        """Return modes >= lower_bound and < upper_bound with amplitudes."""
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
        free_sigma: bool = True,
        temperature: float = 295
    ) -> tuple[float, float]:
        """Fit a selected mode range to the theoretical prediction."""
        fitting_range = self.isolate_mode_range(lower_bound, upper_bound)
        fit = fit_spectrum_to_theory_lmfit(fitting_range, lmax, free_sigma)
        self.kC, reduced_sigma = fit
        self.surface_tension = calc_tension_from_reduced_tension(
            self.r0,
            reduced_sigma,
            self.kC,
            temperature,
        )
        return self.kC, self.surface_tension

    def _to_dict(self, include_arrays=True) -> dict:
        """Convert class attributes to a dict."""
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
        """Save class attributes to JSON."""
        outfile = Path(outfile).with_suffix('.json')
        with outfile.open("w", encoding="utf-8") as f:
            json.dump(self._to_dict(include_arrays=include_arrays), f, indent=indent)
