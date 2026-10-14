#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate and fit vesicle fluctuation spectra.

``Spectrum`` converts accepted vesicle contours into a fluctuation spectrum and
fits a fixed q interval supplied through :class:`SpectrumFitConfig`. Optional or
experimental procedures may choose those bounds before calling the core fit,
but ``Spectrum`` itself does not depend on any range-selection strategy.
"""
from pathlib import Path
import json
import numpy as np
from vesmod.VesEdge import VesicleEdges
from vesmod.validation import require_finite_array, require_numeric_array
from .spectrum_plotting import (
    SpectrumPlotData,
    SpectrumPlotConfig,
    SpectrumPlotResult,
    plot_spectrum,
    save_spectrum_fit_diagnostic,
)
from .config import SpectrumFitConfig
from .calculators import MeanSquaredPowerCalculator
from .power_calculation import PowerCalculationResult, SpectrumCalculator
from .fit_result import SpectrumFit
from .spectrum_utils import (
    MiniSpectrum,
    calc_tension_from_reduced_tension,
    fit_spectrum_lmfit,
)


class Spectrum:
    """Calculate and fit the fluctuation spectrum of one vesicle trajectory.

    Successful construction guarantees a finite positive two-dimensional radii
    trajectory with at least one frame and two angular samples. Temporal
    variance power requires at least two frames. Derived ``r0``,
    ``powers``, ``modes``, and ``fit_results`` state is initialized before
    the object is exposed to callers. Calculated arrays are independent copies; metadata accessors are read-only.

    ``kC`` and ``surface_tension`` are compatibility attributes containing the
    most recent successful physical fit. Durable per-fit provenance is stored
    in ``fit_results``.

    The ``calculator`` callable defines the power calculation; mean-square
    power is the default.
    """

    def __init__(
        self,
        edges_over_time: str | Path | VesicleEdges,
        frame_cutoff=None,
        calculator: SpectrumCalculator | None = None,
    ) -> None:
        """Create a Spectrum from accepted radii or a QCed VesicleEdges object.

        ``calculator`` supplies the contour-to-power calculation.
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

        self._initialize_from_radii(input_data, frame_cutoff, calculator)

    @classmethod
    def from_radii(
        cls,
        radii: np.ndarray,
        frame_cutoff=None,
        calculator: SpectrumCalculator | None = None,
    ) -> "Spectrum":
        """Create a Spectrum directly from an in-memory radii array.

        The caller's array is validated and copied so later mutations do not
        change the constructed spectrum. ``calculator`` defines its powers.
        """
        spectrum = cls.__new__(cls)
        spectrum._initialize_from_radii(radii, frame_cutoff, calculator)
        return spectrum

    @staticmethod
    def _validate_radii(radii: np.ndarray) -> np.ndarray:
        """Return a validated float copy of a contour-radius trajectory."""
        require_numeric_array(radii, "radii")
        if radii.ndim != 2:
            raise ValueError("radii must be a two-dimensional array.")
        if radii.shape[0] == 0:
            raise ValueError("radii must contain at least one frame.")
        if radii.shape[1] < 2:
            raise ValueError("radii must contain at least two angular samples.")
        if np.issubdtype(radii.dtype, np.complexfloating):
            raise TypeError("radii must contain real-valued numbers.")

        require_finite_array(radii, "radii")
        validated = np.asarray(radii, dtype=float).copy()
        if np.any(validated <= 0):
            raise ValueError("radii must contain only positive values.")

        mean_radius = float(np.mean(validated))
        if not np.isfinite(mean_radius) or mean_radius <= 0:
            raise ValueError("radii must have a finite, positive mean radius.")
        return validated

    def _initialize_from_radii(
        self,
        radii: np.ndarray,
        frame_cutoff,
        calculator: SpectrumCalculator | None,
    ) -> None:
        """Validate radii, apply an optional cutoff, and calculate the spectrum."""
        if (
            frame_cutoff is not None
            and (isinstance(frame_cutoff, bool) or not isinstance(frame_cutoff, int))
        ):
            raise TypeError("frame_cutoff must either be None or an int.")
        if isinstance(frame_cutoff, int) and frame_cutoff <= 0:
            raise ValueError("frame_cutoff must be a positive int.")

        input_data = self._validate_radii(radii)
        if frame_cutoff is not None and frame_cutoff < input_data.shape[0]:
            input_data = input_data[:frame_cutoff, :]
        if calculator is None:
            calculator = MeanSquaredPowerCalculator()
        if not callable(calculator):
            raise TypeError("calculator must be callable.")
        result = calculator(input_data.copy())
        if not isinstance(result, PowerCalculationResult):
            raise TypeError("calculator must return a PowerCalculationResult.")
        if result.radius_unit not in {"um", "m", "nm"}:
            raise ValueError("Unsupported radius_unit; use um, m or nm.")
        self._calculation = result
        self.kC = None
        self.surface_tension = None
        self.fit_result = None
        self.fit_results: list[SpectrumFit] = []

    @property
    def powers(self) -> np.ndarray:
        """Independently stored powers aligned with :attr:`modes`."""
        return self._calculation.powers

    @property
    def modes(self) -> np.ndarray:
        """Independently stored signed Fourier modes returned by the calculator."""
        return self._calculation.modes

    @property
    def r0(self) -> float:
        """Calculator reference radius in micrometers."""
        factors = {"um": 1.0, "m": 1e6, "nm": 1e-3}
        try:
            return self._calculation.reference_radius * factors[self._calculation.radius_unit]
        except KeyError as error:
            raise ValueError("Unsupported radius_unit; use um, m or nm.") from error

    @property
    def power_definition(self) -> str:
        """Read-only description of the calculated powers."""
        return self._calculation.power_definition

    def isolate_mode_range(self, lower_bound: int, upper_bound: int) -> MiniSpectrum:
        """Return modes with ``lower_bound <= q < upper_bound`` and amplitudes."""
        if self.modes is None:
            raise AttributeError("There are no modes; Cannot return mode range.")
        mask1 = self.modes >= lower_bound
        mask2 = self.modes < upper_bound
        combined_mask = mask1 & mask2
        return MiniSpectrum(
            self.modes[combined_mask],
            self.powers[combined_mask],
            None,
        )

    def extract_kc_from_fit(
        self,
        config: SpectrumFitConfig | None = None,
    ) -> SpectrumFit:
        """Fit the configured fixed q interval to the theoretical spectrum.

        Range selection is intentionally outside this core method. Callers that
        choose q bounds dynamically should construct a ``SpectrumFitConfig``
        containing those selected bounds before invoking the physical fit.
        """
        if config is None:
            config = SpectrumFitConfig()
        if not isinstance(config, SpectrumFitConfig):
            raise TypeError("config must be a SpectrumFitConfig or None.")

        self.fit_result = None
        fitting_range = self.isolate_mode_range(
            config.lower_bound,
            config.upper_bound,
        )
        if fitting_range.modes.size == 0:
            raise ValueError("Configured q range contains no spectrum modes.")

        self.fit_result = fit_spectrum_lmfit(
            fitting_range,
            config.lmax,
            config.free_sigma,
            config=config, radii=[self.r0], expected_replica_count=1,
        )

        fitted_kc = self.fit_result.best_values["kC"]
        reduced_sigma = self.fit_result.best_values["sigma"]
        fitted_surface_tension = calc_tension_from_reduced_tension(
            self.r0,
            reduced_sigma,
            fitted_kc,
            config.temperature,
        )

        self.kC = fitted_kc
        self.surface_tension = fitted_surface_tension

        fit_result = SpectrumFit(
            kC=float(fitted_kc),
            surface_tension=float(fitted_surface_tension),
            lower_bound=config.lower_bound,
            upper_bound=config.upper_bound,
            config=config,
            power_definition=getattr(self, "power_definition", "mean_square"),
        )

        self.fit_results.append(fit_result)

        return fit_result

    def save_fit_diagnostic(
        self,
        path: str | Path,
        lower_bound: int,
        upper_bound: int,
        lmax: int,
    ) -> None:
        """Save measured spectrum, attempted fit, and residual diagnostics."""
        if self.fit_result is None:
            raise ValueError("A spectrum fit must be attempted before plotting.")
        save_spectrum_fit_diagnostic(
            SpectrumPlotData(
                modes=self.modes,
                avg_amps2=self.powers,
                fit_result=self.fit_result,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lmax=lmax,
            ),
            path,
        )

    def plot(
        self,
        *,
        ax=None,
        color=None,
        label=None,
        fit_config: SpectrumFitConfig | None = None,
        plot_config: SpectrumPlotConfig | None = None,
    ) -> SpectrumPlotResult:
        """Plot this spectrum on a caller-owned or newly created axis.

        A spectrum created from a frame block uses the same method as one
        created from a complete trajectory. When a successful fit exists, its
        recorded configuration supplies the fitting interval unless an explicit
        ``fit_config`` is provided.
        """
        if fit_config is None and self.fit_results:
            fit_config = self.fit_results[-1].config
        fit_result = self.fit_result
        data = SpectrumPlotData(
            modes=self.modes,
            avg_amps2=self.powers,
            fit_result=fit_result,
            lower_bound=None if fit_config is None else fit_config.lower_bound,
            upper_bound=None if fit_config is None else fit_config.upper_bound,
            lmax=None if fit_config is None else fit_config.lmax,
            exposure_time=(fit_config.exposure_time if fit_config is not None
                           and fit_config.camera_integration else 0.0),
        )
        return plot_spectrum(
            data,
            ax=ax,
            color=color,
            label=label,
            config=plot_config,
        )

    def to_dict(self, include_arrays=True) -> dict:
        """Return spectrum state and retained physical-fit provenance."""
        data = {
            "power_definition": getattr(self, "power_definition", "mean_square"),
            "r0": float(self.r0) if getattr(self, "r0", None) is not None else None,
            "kC": float(self.kC) if getattr(self, "kC", None) is not None else None,
            "surface_tension": (
                float(self.surface_tension)
                if getattr(self, "surface_tension", None) is not None
                else None
            ),
        }

        data["reference_radius"] = self._calculation.reference_radius
        data["radius_unit"] = self._calculation.radius_unit
        data["power_unit"] = self._calculation.power_unit
        fit_results = getattr(self, "fit_results", None)
        if fit_results:
            data["fit_results"] = [result.to_dict() for result in fit_results]

        if include_arrays:
            data["modes"] = (
                self.modes.tolist()
                if getattr(self, "modes", None) is not None
                else None
            )
            data["powers"] = (
                self.powers.tolist()
                if getattr(self, "powers", None) is not None
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
            json.dump(self.to_dict(include_arrays=include_arrays), f, indent=indent)
