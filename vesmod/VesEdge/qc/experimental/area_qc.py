"""Contour-area deviation quality control for VesEdge trajectories."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from dataclasses import dataclass

from ..qc_check import QCCheck
from ..qc_result import QCResult
from ...contour_geometry import contour_area
from ...models import EdgeDetection
from vesmod.validation import require_fraction


CHECK_NAME = "area"


@dataclass(frozen=True)
class AreaQCConfig:
    """Configuration owned by the contour-area QC check."""

    max_relative_deviation: float = 0.25
    enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_relative_deviation", require_fraction(self.max_relative_deviation, "max_relative_deviation", include_one=False))
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")


def check_area_deviation(
    detections: list[EdgeDetection],
    max_relative_deviation: float,
) -> QCResult:
    """Evaluate contours against the trajectory median enclosed area.

    A detection fails only when its absolute fractional area deviation is
    strictly greater than max_relative_deviation. A detection exactly on
    either configured bound passes.

    Parameters
    ----------
    detections : list[EdgeDetection]
        Successful detections from one vesicle trajectory.
    max_relative_deviation : float
        Maximum allowed abs(area - median_area) / median_area.

    Returns
    -------
    QCResult
        Areas in pixels squared, fractional deviations, and the reference
        area. Every finite, positive contour area contributes to the median,
        independently of other checks. An unavailable reference is None.

    Raises
    ------
    ValueError
        If no detections are supplied or the threshold is invalid.
    """
    if not detections:
        raise ValueError("Area QC requires at least one successful detection.")
    if (
        not np.isfinite(max_relative_deviation)
        or not 0 <= max_relative_deviation < 1
    ):
        raise ValueError(
            "max_relative_deviation must be finite, at least 0, and less than 1."
        )

    areas = np.asarray(
        [contour_area(edge.full_contour.r) for edge in detections],
        dtype=np.float64,
    )
    reference_candidates = areas[np.isfinite(areas) & (areas > 0)]
    if reference_candidates.size:
        reference_area = float(np.median(reference_candidates))
        deviations = np.abs(areas - reference_area) / reference_area
    else:
        reference_area = float("nan")
        deviations = np.full(areas.shape, np.nan, dtype=np.float64)

    rejected = (
        ~np.isfinite(deviations) | (areas <= 0)
        | ((deviations > max_relative_deviation) & ~np.isclose(
            deviations, max_relative_deviation, rtol=1e-12, atol=1e-15,
        ))
    )
    return QCResult(
        check_name=CHECK_NAME,
        frame_indices=tuple(edge.frame_index for edge in detections),
        rejected=tuple(bool(value) for value in rejected),
        diagnostics={
            "area_pixels2": tuple(areas),
            "relative_area_deviation": tuple(deviations),
        },
        extras={"reference_area_pixels2": reference_area},
    )


def _save_area_qc_plot(path: Path, edges) -> None:
    """Plot contour area by source frame with configured acceptance bounds."""
    import matplotlib.pyplot as plt

    area_result = edges.qc_result.get_result("area")
    config = edges.qc_result.config
    frame_indices = area_result.frame_indices
    areas = area_result.diagnostics["area_pixels2"]
    reference = area_result.extras["reference_area_pixels2"]
    if reference is None:
        reference = float("nan")
    deviation = config.area.max_relative_deviation
    lower_bound = reference * (1 - deviation)
    upper_bound = reference * (1 + deviation)

    figure, axis = plt.subplots()
    axis.plot(frame_indices, areas, ".", color="tab:blue", label="contour area")
    axis.axhline(reference, color="black", label="trajectory median")
    axis.axhline(
        lower_bound,
        color="tab:red",
        linestyle="--",
        label="acceptance bounds",
    )
    axis.axhline(upper_bound, color="tab:red", linestyle="--")
    axis.set_xlabel("Source frame")
    axis.set_ylabel("Contour area (pixels squared)")
    axis.legend()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_area_qc_csv(path: Path, edges) -> None:
    """Write exact per-frame contour-area QC measurements."""
    area_result = edges.qc_result.get_result("area")
    rows = []
    for frame_index, area, deviation, rejected in zip(
        area_result.frame_indices,
        area_result.diagnostics["area_pixels2"],
        area_result.diagnostics["relative_area_deviation"],
        area_result.rejected,
        strict=True,
    ):
        rows.append(
            {
                "frame_index": frame_index,
                "area_pixels2": area,
                "relative_area_deviation": deviation,
                "area_rejected": rejected,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "frame_index",
                "area_pixels2",
                "relative_area_deviation",
                "area_rejected",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


class AreaDeviationCheck(QCCheck):
    """Apply trajectory-wide area-deviation quality control."""

    name = CHECK_NAME
    diagnostic_fields = ('area_pixels2', 'relative_area_deviation')
    extra_result_fields = ('reference_area_pixels2',)
    config_type = AreaQCConfig

    def run(self, detections, config, frames) -> QCResult:
        """Evaluate area deviation across the detection trajectory."""
        del frames
        return check_area_deviation(detections, config.max_relative_deviation)

    def add_cli_arguments(self, parser) -> None:
        """Register command-line options for this check."""
        parser.add_argument(
            "--max-relative-area-deviation",
            type=float,
            default=0.25,
            help=(
                "Maximum absolute fractional deviation from the trajectory median "
                "contour area. Default: 0.25."
            ),
        )
        toggle = parser.add_mutually_exclusive_group()
        toggle.add_argument(
            "--area-qc", dest="area_qc", action="store_true",
            help="Enable experimental contour-area deviation QC.",
        )
        toggle.add_argument(
            "--no-area-qc", dest="area_qc", action="store_false", default=False,
            help="Disable experimental contour-area deviation QC (default).",
        )

    def config_from_args(self, args):
        """Build settings from parsed command-line options."""
        return AreaQCConfig(
            max_relative_deviation=args.max_relative_area_deviation,
            enabled=getattr(args, "area_qc", False),
        )

    summary_order = 1
    diagnostic_writers = (
        ('.area_qc.png', _save_area_qc_plot),
        ('.area_qc.csv', _write_area_qc_csv),
    )
