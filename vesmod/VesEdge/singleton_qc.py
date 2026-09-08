"""Geometry-only QC for isolated radial contour excursions."""

import numpy as np

from .contour_geometry import fit_radial_baseline
from .models import EdgeDetection, QCFlag


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
