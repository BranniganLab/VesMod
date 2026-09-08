"""Frame-level QC for malformed or collapsed extracted contours."""

import numpy as np

from .models import EdgeDetection, QCFlag


def check_minimum_radius(edge: EdgeDetection, min_median_radius_pixels: float) -> None:
    """Reject contours containing invalid or exceptionally small radii.

    The median native contour radius is recorded for diagnostics. Any
    nonfinite or nonpositive sample fails immediately; otherwise a contour
    fails when its median radius is below the configured pixel threshold.
    """
    if not np.isfinite(min_median_radius_pixels) or min_median_radius_pixels < 0:
        raise ValueError("min_median_radius_pixels must be finite and non-negative.")
    radii = np.asarray(edge.full_contour.r, dtype=float)
    median = float(np.median(radii)) if radii.size else float("nan")
    edge.qc.diagnostics["median_radius_pixels"] = median
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        edge.qc.flags.add(QCFlag.MINIMUM_RADIUS)
    elif median < min_median_radius_pixels:
        edge.qc.flags.add(QCFlag.MINIMUM_RADIUS)
    else:
        edge.qc.flags.discard(QCFlag.MINIMUM_RADIUS)
