"""Fit-result records for EdgeMod spectrum analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .config import SpectrumFitConfig
from .fit_range_selection import FitRangeSelection


@dataclass(frozen=True)
class SpectrumFit:
    """Store one physical fit of a Spectrum.

    Parameters
    ----------
    kC : float
        Fitted bending modulus.
    surface_tension : float
        Surface tension converted from the fitted reduced tension.
    lower_bound : int
        Inclusive lower q bound used for the physical fit.
    upper_bound : int
        Exclusive upper q bound used for the physical fit.
    config : SpectrumFitConfig
        Scientific configuration used to produce this fit.
    range_selection : FitRangeSelection | None
        Range-selection diagnostics associated with the fit.
    """

    kC: float
    surface_tension: float
    lower_bound: int
    upper_bound: int
    config: SpectrumFitConfig
    range_selection: FitRangeSelection | None = None

    @property
    def method(self) -> str:
        """Return the class name of the configured q-range selector."""
        return type(self.config.range_selector).__name__

    def __iter__(self) -> Iterator[float]:
        """Preserve ``kc, tension = extract_kc_from_fit(...)`` unpacking."""
        yield self.kC
        yield self.surface_tension

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the fit."""
        data = {
            "kC": float(self.kC),
            "surface_tension": float(self.surface_tension),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "method": self.method,
            "config": self.config.to_dict(),
        }
        if self.range_selection is not None:
            data["range_selection"] = self.range_selection.to_dict()
        return data
