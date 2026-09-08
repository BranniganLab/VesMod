"""Frame-level QC for malformed or collapsed extracted contours."""

from dataclasses import dataclass

import numpy as np

from .models import EdgeDetection, QCFlag
from vesmod.validation import require_nonnegative_real


@dataclass(frozen=True)
class MinimumRadiusQCConfig:
    """Configuration owned by the minimum-radius QC check."""

    enabled: bool = False
    min_median_radius_pixels: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        object.__setattr__(self, "min_median_radius_pixels", require_nonnegative_real(self.min_median_radius_pixels, "min_median_radius_pixels"))


@dataclass(frozen=True)
class MinimumRadiusQCResult:
    """Per-trajectory outcome produced by the minimum-radius QC check."""

    median_radii_pixels: tuple[float, ...]
    rejected_count: int


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
