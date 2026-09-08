"""Geometry-only QC against a low-order radial baseline."""

from dataclasses import dataclass

import numpy as np
from .models import EdgeDetection, QCFlag
from .contour_geometry import fit_radial_baseline
from vesmod.validation import require_integer_valued, require_nonnegative_real


@dataclass(frozen=True)
class LocalizedDeviationQCConfig:
    """Configuration owned by the localized-deviation QC check."""

    enabled: bool = False
    order: int = 3
    max_residual_fraction: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        order = require_integer_valued(self.order, "order")
        if order < 0:
            raise ValueError("order must be non-negative.")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "max_residual_fraction", require_nonnegative_real(self.max_residual_fraction, "max_residual_fraction"))


def check_localized_deviation(edge: EdgeDetection, order: int, max_residual_fraction: float) -> None:
    """Reject contours whose largest baseline residual is excessive."""
    if order < 0 or not np.isfinite(max_residual_fraction) or max_residual_fraction < 0:
        raise ValueError("localized-deviation QC configuration must use finite non-negative values")
    radii = np.asarray(edge.analysis_contour.r, dtype=float)
    median = float(np.median(radii))
    fitted = fit_radial_baseline(radii, order).values
    score = float(np.max(np.abs(radii - fitted)) / median)
    edge.qc.diagnostics["localized_deviation_score"] = score
    if score > max_residual_fraction:
        edge.qc.flags.add(QCFlag.LOCALIZED_DEVIATION)
    else:
        edge.qc.flags.discard(QCFlag.LOCALIZED_DEVIATION)
