"""Immutable fit-result records for EdgeMod spectrum analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .config import SpectrumFitConfig
from .fit_range_selection import FitRangeSelection


@dataclass(frozen=True)
class SpectrumFit:
    """Store one physical fit of a :class:`Spectrum` without overwriting others.

    A ``Spectrum`` may be fit repeatedly with different fixed or dynamic q-range
    strategies. Each successful fit is represented by its own immutable
    ``SpectrumFit`` so the fitted values, actual q bounds, scientific config,
    and range-selection diagnostics remain associated with one another.

    Parameters
    ----------
    kC : float
        Fitted bending modulus.
    surface_tension : float
        Surface tension converted from the fitted reduced tension.
    lower_bound : int
        Inclusive lower q bound actually used for the physical fit.
    upper_bound : int
        Exclusive upper q bound actually used for the physical fit.
    config : SpectrumFitConfig
        Scientific configuration used to produce this fit.
    range_selection : FitRangeSelection | None
        Selector result describing how the q range was chosen. Dynamic
        selectors additionally report slope, log-space RMSE, and rejection
        information. Successful physical fits always have an accepted
        selection.
    """

    kC: float
    surface_tension: float
    lower_bound: int
    upper_bound: int
    config: SpectrumFitConfig
    range_selection: FitRangeSelection | None = None

    @property
    def method(self) -> str:
        """Return the class name of the q-range selector that produced the fit."""
        return type(self.config.range_selector).__name__

    def __iter__(self) -> Iterator[float]:
        """Allow compatibility unpacking as ``kc, tension = fit``."""
        yield self.kC
        yield self.surface_tension

    def to_dict(self) -> dict:
        """Return fit values, q bounds, config, and diagnostics for JSON output."""
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
