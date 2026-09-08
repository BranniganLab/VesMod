"""Geometry-only QC against a low-order radial baseline."""

import numpy as np
from .models import EdgeDetection, QCFlag
from .contour_geometry import fit_radial_baseline


def check_localized_deviation(edge: EdgeDetection, order: int, max_residual_fraction: float, support_residual_fraction: float | None = None, max_support_samples: int | None = None) -> None:
    """Reject contours whose largest baseline residual is excessive."""
    if order < 0 or not np.isfinite(max_residual_fraction) or max_residual_fraction < 0:
        raise ValueError("localized-deviation QC configuration must use finite non-negative values")
    if (support_residual_fraction is None) != (max_support_samples is None):
        raise ValueError("support residual and maximum support must be configured together")
    radii = np.asarray(edge.analysis_contour.r, dtype=float)
    median = float(np.median(radii))
    fitted = fit_radial_baseline(radii, order).values
    score = float(np.max(np.abs(radii - fitted)) / median)
    edge.qc.localized_deviation_score = score
    rejected = score > max_residual_fraction
    support = None
    if support_residual_fraction is not None:
        residual = np.abs(radii - fitted) / median
        mask = np.concatenate((residual >= support_residual_fraction, residual >= support_residual_fraction))
        if np.all(mask):
            support = radii.size
        else:
            lengths = []
            start = None
            for index, value in enumerate(mask):
                if value and start is None:
                    start = index
                elif not value and start is not None:
                    lengths.append(index - start)
                    start = None
            support = min(max(lengths, default=0), radii.size)
        rejected = rejected or (score >= support_residual_fraction and support <= max_support_samples)
    edge.qc.localized_deviation_support_samples = support
    if rejected:
        edge.qc.flags.add(QCFlag.LOCALIZED_DEVIATION)
    else:
        edge.qc.flags.discard(QCFlag.LOCALIZED_DEVIATION)
