#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute vesicle fluctuation spectra and fit membrane mechanical parameters."""

from pathlib import Path
from types import NoneType
import json
import numpy as np
from vesmod.VesEdge import VesicleEdges
from .config import SpectrumFitConfig
from .fit_range_selection import FitRangeSelection
from .fit_result import SpectrumFit
from .spectrum_utils import (
    MiniSpectrum,
    calc_tension_from_reduced_tension,
    fit_spectrum_to_theory_lmfit,
)


class Spectrum:
    """Represent one measured vesicle fluctuation spectrum and its fit history.

    ``Spectrum`` stores the measured Fourier modes and mean-squared amplitudes.
    Scientific fitting choices are supplied separately through
    :class:`SpectrumFitConfig`, allowing the same measured spectrum to be fit
    repeatedly with fixed or dynamically selected q ranges. Successful fits are
    retained in ``fit_results`` rather than replacing one another.

    ``kC`` and ``surface_tension`` remain convenience attributes containing the
    most recent successful fit values.
    """

    def __init__(
        self,
        edges_over_time: str | Path | VesicleEdges,
        frame_cutoff=None
    ) -> None:
        """Create a spectrum from accepted contour radii.

        Parameters
        ----------
        edges_over_time : str | Path | VesicleEdges
            A ``.npy`` trajectory of contour radii in microns or a QC-completed
            ``VesicleEdges`` object. Only accepted detections from
            ``VesicleEdges`` contribute to the spectrum.
        frame_cutoff : int | None, optional
            If provided, use only the first ``frame_cutoff`` contour frames.
        """
        if isinstance(edges_over_time, VesicleEdges):
            input_data = edges_over_time.accepted_radii_microns
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
        self.fit_range_selection: FitRangeSelection | None = None
        self.fit_results: list[SpectrumFit] = []

    def _calc_avg_sq_amplitudes(self, r_vals_over_time: np.ndarray) -> np.ndarray:
        """Return frame-averaged squared Fourier amplitudes normalized by r0."""
        n_samples = r_vals_over_time.shape[1]
        norm = 1. / (self.r0 * n_samples)
        amps = np.fft.fft(r_vals_over_time, axis=1, norm='backward') * norm
        amps2 = amps * amps.conj()
        avg_amps2 = np.mean(amps2.real, axis=0)
        return avg_amps2

    def _calc_integer_modes(self) -> np.ndarray[int]:
        """Return integer Fourier mode numbers in NumPy FFT ordering."""
        freqs = np.fft.fftfreq(self.avg_amps2.shape[0])
        modes = np.round(freqs * self.avg_amps2.shape[0]).astype(int)
        return modes

    def isolate_mode_range(self, lower_bound: int, upper_bound: int) -> MiniSpectrum:
        """Return modes with ``lower_bound <= q < upper_bound`` and amplitudes."""
        if self.modes is None:
            raise AttributeError("There are no modes; Cannot return mode range.")
        mask1 = self.modes >= lower_bound
        mask2 = self.modes < upper_bound
        combined_mask = mask1 & mask2
        return MiniSpectrum(
            self.modes[combined_mask],
            self.avg_amps2[combined_mask],
            None,
        )

    def extract_kc_from_fit(
        self,
        config: SpectrumFitConfig | None = None,
    ) -> SpectrumFit:
        """Select a q range and fit that range to the theoretical spectrum.

        Parameters
        ----------
        config : SpectrumFitConfig | None
            Scientific fit configuration. If omitted, use the historical
            default fixed fit over q = 3, 4, 5, 6, 7 with ``lmax=500``, free
            surface tension, and ``temperature=295 K``.

        Returns
        -------
        SpectrumFit
            Immutable fit result containing fitted values, the actual q bounds
            used, the full configuration, and range-selection diagnostics. The
            result is appended to ``fit_results``.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SpectrumFitConfig`` or ``None``.
        ValueError
            If the configured range selector rejects the spectrum or returns an
            accepted selection without q bounds. Dynamic rejection occurs
            before the physical spectrum fit is run.
        """
        if config is None:
            config = SpectrumFitConfig()
        if not isinstance(config, SpectrumFitConfig):
            raise TypeError("config must be a SpectrumFitConfig or None.")

        selection = config.range_selector.select(
            self.modes,
            self.avg_amps2,
        )
        self.fit_range_selection = selection
        if not selection.accepted:
            self.kC = None
            self.surface_tension = None
            raise ValueError(selection.reason or "No acceptable q range found.")
        if selection.lower_bound is None or selection.upper_bound is None:
            raise ValueError("Accepted fit-range selection is missing q bounds.")

        fitting_range = self.isolate_mode_range(
            selection.lower_bound,
            selection.upper_bound,
        )
        self.kC, reduced_sigma = fit_spectrum_to_theory_lmfit(
            fitting_range,
            config.lmax,
            config.free_sigma,
        )
        self.surface_tension = calc_tension_from_reduced_tension(
            self.r0,
            reduced_sigma,
            self.kC,
            config.temperature,
        )
        fit_result = SpectrumFit(
            kC=float(self.kC),
            surface_tension=float(self.surface_tension),
            lower_bound=selection.lower_bound,
            upper_bound=selection.upper_bound,
            config=config,
            range_selection=selection,
        )
        if not hasattr(self, "fit_results"):
            self.fit_results = []
        self.fit_results.append(fit_result)
        return fit_result

    def _to_dict(self, include_arrays=True) -> dict:
        """Return spectrum state and retained fit results as serializable data."""
        data = {
            "r0": float(self.r0) if getattr(self, "r0", None) is not None else None,
            "kC": float(self.kC) if getattr(self, "kC", None) is not None else None,
            "surface_tension": (
                float(self.surface_tension)
                if getattr(self, "surface_tension", None) is not None
                else None
            ),
        }

        selection = getattr(self, "fit_range_selection", None)
        if selection is not None:
            data["fit_range_selection"] = selection.to_dict()

        fit_results = getattr(self, "fit_results", None)
        if fit_results:
            data["fit_results"] = [result.to_dict() for result in fit_results]

        if include_arrays:
            data["modes"] = (
                self.modes.tolist()
                if getattr(self, "modes", None) is not None
                else None
            )
            data["avg_amps2"] = (
                self.avg_amps2.tolist()
                if getattr(self, "avg_amps2", None) is not None
                else None
            )

        return data

    def to_json(
        self,
        outfile: str | Path,
        include_arrays: bool = True,
        indent: int = 2,
    ) -> None:
        """Write spectrum data and all retained fit results to a JSON file."""
        outfile = Path(outfile).with_suffix('.json')
        with outfile.open("w", encoding="utf-8") as f:
            json.dump(self._to_dict(include_arrays=include_arrays), f, indent=indent)
