"""Immutable fit-result records for core EdgeMod spectrum analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .config import SpectrumFitConfig


@dataclass(frozen=True)
class SpectrumFit:
    """Store one physical fit of a :class:`Spectrum` without overwriting others.

    A ``Spectrum`` may be fit repeatedly over different q ranges. Each
    successful fit is represented by its own immutable ``SpectrumFit`` so the
    fitted values and exact physical-fit configuration remain associated.

    Experimental procedures that choose q bounds retain their own selection
    diagnostics separately from this core physical-fit result.

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
        Scientific configuration used to produce this physical fit.
    """

    kC: float
    surface_tension: float
    lower_bound: int
    upper_bound: int
    config: SpectrumFitConfig

    def __iter__(self) -> Iterator[float]:
        """Allow compatibility unpacking as ``kc, tension = fit``."""
        yield self.kC
        yield self.surface_tension

    def to_dict(self) -> dict:
        """Return fit values, q bounds, and config for JSON output."""
        return {
            "kC": float(self.kC),
            "surface_tension": float(self.surface_tension),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "config": self.config.to_dict(),
        }
