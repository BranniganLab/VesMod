"""Geometry-only QC for isolated radial contour excursions."""

from dataclasses import dataclass

import numpy as np

from .contour_geometry import fit_radial_baseline
from .models import EdgeDetection, QCFlag
from vesmod.validation import require_integer_valued, require_nonnegative_real


@dataclass(frozen=True)
class SingletonDeviationQCConfig:
    """Configuration owned by the singleton-deviation QC check."""

    enabled: bool = False
    order: int = 3
    min_residual_fraction: float = 0.05
    max_width_samples: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        order = require_integer_valued(self.order, "order")
        width = require_integer_valued(self.max_width_samples, "max_width_samples")
        if order < 0:
            raise ValueError("order must be non-negative.")
        if width <= 0:
            raise ValueError("max_width_samples must be positive.")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "max_width_samples", width)
        object.__setattr__(self, "min_residual_fraction", require_nonnegative_real(self.min_residual_fraction, "min_residual_fraction"))


@dataclass(frozen=True)
class SingletonDeviationQCResult:
    """Per-trajectory outcome produced by singleton-deviation QC."""

    scores: tuple[float, ...]
    counts: tuple[int, ...]
    rejected_count: int


def check_singleton_deviation(
    edge: EdgeDetection,
    order: int,
    min_residual_fraction: float,
    max_width_samples: int,
) -> None:
    """Reject contours containing narrow excursions from a smooth baseline."""
    radii = np.asarray(edge.analysis_contour.r, dtype=float)
    median = float(np.median(radii))
    baseline = fit_radial_baseline(radii, order).values
    residual = np.abs(radii - baseline) / median
    above = residual >= min_residual_fraction
    # Count contiguous threshold crossings on a circular contour.
    doubled = np.concatenate((above, above))
    runs = []
    start = None
    for index, value in enumerate(doubled):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(doubled) - start))
    widths = [width for start, width in runs if start < radii.size and start + width > radii.size]
    widths.extend(width for start, width in runs if start < radii.size and start + width <= radii.size)
    singleton_count = sum(width <= max_width_samples for width in widths)
    edge.qc.diagnostics["singleton_count"] = singleton_count
    edge.qc.diagnostics["singleton_score"] = float(np.max(residual)) if residual.size else 0.0
    if singleton_count:
        edge.qc.flags.add(QCFlag.SINGLETON_DEVIATION)
    else:
        edge.qc.flags.discard(QCFlag.SINGLETON_DEVIATION)
