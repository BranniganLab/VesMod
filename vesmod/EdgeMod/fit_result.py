"""Fit-result records for EdgeMod spectrum analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

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
    method : str
        Human-readable identifier for how the fit range was chosen.
    range_selection : FitRangeSelection | None
        Dynamic range-selection diagnostics, when applicable.
    """

    kC: float
    surface_tension: float
    lower_bound: int
    upper_bound: int
    method: str
    range_selection: FitRangeSelection | None = None

    def __iter__(self) -> Iterator[float]:
        """Preserve legacy ``kc, tension = extract_kc_from_fit(...)`` unpacking."""
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
        }
        if self.range_selection is not None:
            data["range_selection"] = self.range_selection.to_dict()
        return data
