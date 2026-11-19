"""Replica spectra and fit provenance retained by an ensemble."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from vesmod.edgemod.fitting.config import SpectrumFitConfig
from vesmod.edgemod.fitting.fit_result import SpectrumFit
from vesmod.edgemod.spectra.power_calculation.base import PowerCalculationResult


@dataclass(frozen=True, eq=False)
class EnsembleMember:
    """Measured powers and optional fit history for one replica.

    Parameters
    ----------
    powers, modes : array-like
        Measured powers and their corresponding Fourier modes.
    kC, surface_tension : float or None
        Summary values from a saved spectrum without a selected fit.
    r0 : float or None
        Reference radius in micrometers.
    power_definition, power_unit : str
        Definition and unit of the measured powers.
    source : pathlib.Path or None
        Source identity, typically the saved spectrum path.
    fit_results : tuple[SpectrumFit, ...]
        Fit history in chronological order. The last fit is selected.
    """

    powers: np.ndarray
    modes: np.ndarray
    kC: float | None = None
    r0: float | None = None
    power_definition: str = "mean_square"
    power_unit: str = "dimensionless"
    source: Path | None = None
    fit_results: tuple[SpectrumFit, ...] = ()
    surface_tension: float | None = None

    def __post_init__(self):
        """Validate measured powers and bind the summary to the selected fit."""
        calculation = PowerCalculationResult(
            modes=self.modes, powers=self.powers,
            reference_radius=1.0 if self.r0 is None else self.r0,
            radius_unit="um", power_definition=self.power_definition,
            power_unit=self.power_unit,
        )
        for name in ("modes", "powers"):
            array = getattr(calculation, name)
            object.__setattr__(self, name, np.frombuffer(array.tobytes(), dtype=array.dtype))
        if self.r0 is not None:
            object.__setattr__(self, "r0", calculation.reference_radius)
        fits = tuple(self.fit_results)
        if any(not isinstance(fit, SpectrumFit) for fit in fits):
            raise TypeError("fit_results must contain SpectrumFit records.")
        object.__setattr__(self, "fit_results", fits)
        if fits:
            object.__setattr__(self, "kC", fits[-1].kC)
            object.__setattr__(self, "surface_tension", fits[-1].surface_tension)

    @classmethod
    def from_spectrum(cls, spectrum, *, source=None):
        """Create a member from a calculated spectrum.

        Parameters
        ----------
        spectrum : Spectrum
            Measured spectrum and its optional fit history.
        source : pathlib.Path or None
            Identity to retain with the member.

        Returns
        -------
        EnsembleMember
            Replica record ready for aggregation.
        """
        return cls(
            powers=spectrum.powers,
            modes=spectrum.modes,
            kC=spectrum.kC,
            r0=spectrum.r0,
            power_definition=spectrum.power_definition,
            power_unit=spectrum.power_unit,
            source=source,
            fit_results=tuple(spectrum.fit_results),
            surface_tension=spectrum.surface_tension,
        )

    @classmethod
    def from_json(cls, path):
        """Load a member from a per-vesicle spectrum export.

        Parameters
        ----------
        path : str or pathlib.Path
            Saved spectrum JSON with the measured arrays included.

        Returns
        -------
        EnsembleMember
            Validated spectrum with the saved fit history and source path.
        """
        path = Path(path)
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
        if not isinstance(data, dict) or not {"modes", "powers", "r0"} <= data.keys():
            raise ValueError("Expected a per-vesicle spectrum JSON with modes, powers, and r0.")
        if data["r0"] is None:
            raise ValueError("A saved spectrum must supply r0.")
        fits = []
        for fit in data.get("fit_results", []):
            if not isinstance(fit, dict) or not isinstance(fit.get("config"), dict):
                raise ValueError("Expected a saved spectrum fit with a configuration.")
            fits.append(SpectrumFit(**{**fit, "config": SpectrumFitConfig(**fit["config"])}))
        return cls(
            powers=data["powers"],
            modes=data["modes"],
            kC=data.get("kC"),
            r0=data["r0"],
            power_definition=data.get("power_definition"),
            power_unit=data.get("power_unit"),
            source=path.resolve(),
            fit_results=tuple(fits),
            surface_tension=data.get("surface_tension"),
        )
