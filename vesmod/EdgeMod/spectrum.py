#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate and fit vesicle fluctuation spectra.

``Spectrum`` converts accepted vesicle contours into a fluctuation spectrum and
fits a q range selected by a :class:`SpectrumFitConfig`. Fixed and dynamic
range-selection strategies share the same fitting path. Each successful fit is
retained as an immutable :class:`SpectrumFit` so multiple analyses of one
spectrum can be compared without overwriting earlier results.
"""
from pathlib import Path
from types import NoneType
import json
import numpy as np
from vesmod.VesEdge import VesicleEdges
from .diagnostic_plotting import (
    SpectrumDiagnosticData,
    save_spectrum_fit_diagnostic,
)
from .config import SpectrumFitConfig
from .fit_range_selection import FitRangeSelection
from .fit_result import SpectrumFit
from .spectrum_utils import (
    MiniSpectrum,
    calc_tension_from_reduced_tension,
    fit_spectrum_lmfit,
    validate_lmfit_result,
)


class Spectrum:
    """Calculate and fit the fluctuation spectrum of one vesicle trajectory.

    ``kC`` and ``surface_tension`` are compatibility attributes containing the
    most recent *successful* physical fit. Range-selection failures update
    ``fit_range_selection`` but leave those latest-successful values unchanged.
    Durable per-fit provenance is stored in ``fit_results``.
    """

    def __init__(
        self,
        edges_over_time: str | Path | VesicleEdges,
        frame_cutoff=None
    ) -> None:
        """Create a Spectrum from accepted radii or a QCed VesicleEdges object."""
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
        self.fit_result = None
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
        """Select a q range and fit that range to the theoretical spectrum."""
        if config is None:
            config = SpectrumFitConfig()
        if not isinstance(config, SpectrumFitConfig):
            raise TypeError("config must be a SpectrumFitConfig or None.")

        # Prevent a failed new attempt from exposing an old lmfit result as though
        # it belonged to the current attempt. Do not clear kC/surface_tension:
        # those intentionally retain the most recent successful physical fit.
        self.fit_result = None

        selection = config.range_selector.select(
            self.modes,
            self.avg_amps2,
        )
        self.fit_range_selection = selection

        if not selection.accepted:
            raise ValueError(selection.reason or "No acceptable q range found.")

        if selection.lower_bound is None or selection.upper_bound is None:
            raise ValueError(
                "Accepted fit-range selection is missing q bounds."
            )

        fitting_range = self.isolate_mode_range(
            selection.lower_bound,
            selection.upper_bound,
        )

        # Keep the complete lmfit result so branch 73 can make diagnostics even
        # when validation rejects the physical fit.
        self.fit_result = fit_spectrum_lmfit(
            fitting_range,
            config.lmax,
            config.free_sigma,
        )
        validate_lmfit_result(
            self.fit_result,
            fitting_range,
            config.free_sigma,
        )

        fitted_kc = self.fit_result.best_values["kC"]
        reduced_sigma = self.fit_result.best_values["sigma"]
        fitted_surface_tension = calc_tension_from_reduced_tension(
            self.r0,
            reduced_sigma,
            fitted_kc,
            config.temperature,
        )

        # Only replace compatibility attributes after a successful validated fit.
        self.kC = fitted_kc
        self.surface_tension = fitted_surface_tension

        fit_result = SpectrumFit(
            kC=float(fitted_kc),
            surface_tension=float(fitted_surface_tension),
            lower_bound=selection.lower_bound,
            upper_bound=selection.upper_bound,
            config=config,
            range_selection=selection,
        )

        if not hasattr(self, "fit_results"):
            self.fit_results = []
        self.fit_results.append(fit_result)

        return fit_result

    def save_fit_diagnostic(
        self,
        path: str | Path,
        lower_bound: int,
        upper_bound: int,
        lmax: int,
        validation_error: str | None = None,
    ) -> None:
        """Save measured spectrum, attempted fit, and residual diagnostics."""
        if self.fit_result is None:
            raise ValueError("A spectrum fit must be attempted before plotting.")
        save_spectrum_fit_diagnostic(
            SpectrumDiagnosticData(
                modes=self.modes,
                avg_amps2=self.avg_amps2,
                fit_result=self.fit_result,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lmax=lmax,
                validation_error=validation_error,
            ),
            path,
        )

    def _to_dict(self, include_arrays=True) -> dict:
        """Return spectrum state and retained fit provenance as a dictionary."""
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
        """Serialize spectrum state and retained fit records to JSON."""
        outfile = Path(outfile).with_suffix('.json')
        with outfile.open("w", encoding="utf-8") as f:
            json.dump(self._to_dict(include_arrays=include_arrays), f, indent=indent)
