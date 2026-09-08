"""Geometry-only QC against a low-order radial baseline."""

import numpy as np
from .models import EdgeDetection, QCFlag


def check_baseline(edge: EdgeDetection, order: int, max_residual_fraction: float) -> None:
    """Reject contours whose largest baseline residual is excessive."""
    if order < 0 or not np.isfinite(max_residual_fraction) or not 0 <= max_residual_fraction < 1:
        raise ValueError("invalid baseline QC configuration")
    radii = np.asarray(edge.analysis_contour.r, dtype=float)
    median = float(np.median(radii))
    theta = np.linspace(0.0, 2.0 * np.pi, radii.size, endpoint=False)
    columns = [np.ones(radii.size)]
    for harmonic in range(1, order + 1):
        columns.extend((np.cos(harmonic * theta), np.sin(harmonic * theta)))
    design = np.column_stack(columns)
    fitted = design @ np.linalg.lstsq(design, radii, rcond=None)[0]
    score = float(np.max(np.abs(radii - fitted)) / median)
    edge.qc.baseline_score = score
    if score > max_residual_fraction:
        edge.qc.flags.add(QCFlag.BASELINE)
    else:
        edge.qc.flags.discard(QCFlag.BASELINE)
