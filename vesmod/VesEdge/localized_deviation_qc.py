"""Geometry-only QC against a low-order radial baseline."""

import numpy as np
from .models import EdgeDetection, QCFlag
from .contour_geometry import fit_radial_baseline


def check_localized_deviation(edge: EdgeDetection, order: int, max_residual_fraction: float) -> None:
    """Reject contours whose largest baseline residual is excessive."""
    if order < 0 or not np.isfinite(max_residual_fraction) or max_residual_fraction < 0:
        raise ValueError("localized-deviation QC configuration must use finite non-negative values")
    radii = np.asarray(edge.analysis_contour.r, dtype=float)
    median = float(np.median(radii))
    fitted = fit_radial_baseline(radii, order).values
    score = float(np.max(np.abs(radii - fitted)) / median)
    edge.qc.localized_deviation_score = score
    if score > max_residual_fraction:
        edge.qc.flags.add(QCFlag.LOCALIZED_DEVIATION)
    else:
        edge.qc.flags.discard(QCFlag.LOCALIZED_DEVIATION)
