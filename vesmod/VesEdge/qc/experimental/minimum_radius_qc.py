"""Frame-level QC for malformed or collapsed extracted contours."""

import csv
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from ..qc_check import QCCheck
from ...models import EdgeDetection
from vesmod.validation import require_nonnegative_real


CHECK_NAME = "minimum_radius"


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
        edge.qc.flags.add(CHECK_NAME)
    elif median < min_median_radius_pixels:
        edge.qc.flags.add(CHECK_NAME)
    else:
        edge.qc.flags.discard(CHECK_NAME)


def _write_minimum_radius_qc_csv(path: Path, edges) -> None:
    """Write exact per-frame contour-radius QC measurements."""
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "frame_index",
                "median_radius_pixels",
                "minimum_radius_rejected",
            ],
        )
        writer.writeheader()
        for detection in edges.successful_detections:
            writer.writerow(
                {
                    "frame_index": detection.frame_index,
                    "median_radius_pixels": detection.qc.diagnostics.get("median_radius_pixels"),
                    "minimum_radius_rejected": (
                        CHECK_NAME in detection.qc.flags
                    ),
                }
            )


class MinimumRadiusCheck(QCCheck):
    """Apply the configured minimum-radius check to each detection."""

    name = CHECK_NAME
    config_type = MinimumRadiusQCConfig
    def run(self, detections, config, frames) -> MinimumRadiusQCResult:
        """Apply the minimum-radius check to each detection."""
        del frames
        for detection in detections:
            check_minimum_radius(
                detection,
                min_median_radius_pixels=config.min_median_radius_pixels,
            )
        return MinimumRadiusQCResult(
            median_radii_pixels=tuple(
                float(detection.qc.diagnostics["median_radius_pixels"])
                for detection in detections
            ),
            rejected_count=sum(
                CHECK_NAME in detection.qc.flags
                for detection in detections
            ),
        )

    def add_cli_arguments(self, parser) -> None:
        """Register command-line options for this check."""
        parser.add_argument(
            "--min-median-radius-pixels",
            type=float,
            default=5.0,
            help=(
                "Reject frames whose median native contour radius is below this "
                "value. Default: 5."
            ),
        )
        parser.add_argument(
            "--minimum-radius-qc",
            action="store_true",
            help="Enable frame-level nonpositive/collapsed contour-radius QC.",
        )

    def config_from_args(self, args):
        """Build settings from parsed command-line options."""
        return MinimumRadiusQCConfig(
            enabled=getattr(args, "minimum_radius_qc", False),
            min_median_radius_pixels=getattr(args, "min_median_radius_pixels", 5.0),
        )

    summary_order = 2
    diagnostic_writers = (
        ('.minimum_radius_qc.csv', _write_minimum_radius_qc_csv),
    )
