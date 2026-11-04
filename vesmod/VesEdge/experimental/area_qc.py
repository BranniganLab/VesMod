"""Contour-area deviation quality control for VesEdge trajectories."""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from dataclasses import dataclass

from ..qc.contract import QCCheck
from ..contour_geometry import contour_area
from ..models import EdgeDetection, QCFlag
from vesmod.validation import require_fraction


@dataclass(frozen=True)
class AreaQCConfig:
    """Configuration owned by the contour-area QC check."""

    max_relative_deviation: float = 0.25
    enabled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_relative_deviation", require_fraction(self.max_relative_deviation, "max_relative_deviation", include_one=False))
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")


@dataclass(frozen=True)
class AreaQCResult:
    """Per-trajectory outcome produced by the contour-area QC check."""

    areas_pixels2: tuple[float, ...]
    reference_area_pixels2: float
    relative_deviations: tuple[float, ...]
    rejected_count: int


def check_area_deviation(
    detections: list[EdgeDetection],
    max_relative_deviation: float,
) -> AreaQCResult:
    """Flag contours that deviate from the trajectory median enclosed area.

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
    AreaQCResult
        Areas, trajectory reference, deviations, and rejection count. The
        reference area and deviations are ``nan`` when no curvature-passing
        contour has a finite, positive area; non-finite and nonpositive areas
        are excluded from the reference candidates.

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
    eligible = np.asarray(
        [QCFlag.CURVATURE not in edge.qc.flags for edge in detections],
        dtype=bool,
    )
    reference_candidates = areas[
        eligible & np.isfinite(areas) & (areas > 0)
    ]
    if reference_candidates.size:
        reference_area = float(np.median(reference_candidates))
        deviations = np.abs(areas - reference_area) / reference_area
    else:
        reference_area = float("nan")
        deviations = np.full(areas.shape, np.nan, dtype=np.float64)

    for edge, area, deviation, is_eligible in zip(
        detections,
        areas,
        deviations,
        eligible,
        strict=True,
    ):
        edge.qc.diagnostics["area_pixels2"] = float(area)
        edge.qc.diagnostics["relative_area_deviation"] = float(deviation)
        if not is_eligible:
            edge.qc.flags.discard(QCFlag.AREA_DEVIATION)
            continue
        exceeds_threshold = (
            deviation > max_relative_deviation
            and not np.isclose(
                deviation,
                max_relative_deviation,
                rtol=1e-12,
                atol=1e-15,
            )
        )
        if not np.isfinite(deviation) or exceeds_threshold:
            edge.qc.flags.add(QCFlag.AREA_DEVIATION)
        else:
            edge.qc.flags.discard(QCFlag.AREA_DEVIATION)

    rejected_count = sum(
        QCFlag.AREA_DEVIATION in edge.qc.flags
        for edge in detections
    )
    return AreaQCResult(
        areas_pixels2=tuple(float(area) for area in areas),
        reference_area_pixels2=reference_area,
        relative_deviations=tuple(
            float(deviation)
            for deviation in deviations
        ),
        rejected_count=rejected_count,
    )


def _save_area_qc_plot(path: Path, edges) -> None:
    """Plot contour area by source frame with configured acceptance bounds."""
    import matplotlib.pyplot as plt

    area_result = edges.qc_result.get_result("area")
    config = edges.qc_result.config
    detections = edges.successful_detections
    frame_indices = [edge.frame_index for edge in detections]
    areas = area_result.areas_pixels2
    reference = area_result.reference_area_pixels2
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
    for detection, area, deviation in zip(
        edges.successful_detections,
        area_result.areas_pixels2,
        area_result.relative_deviations,
        strict=True,
    ):
        rows.append(
            {
                "frame_index": detection.frame_index,
                "area_pixels2": area,
                "relative_area_deviation": deviation,
                "area_rejected": (
                    QCFlag.AREA_DEVIATION in detection.qc.flags
                ),
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

    name = "area"
    config_type = AreaQCConfig
    requires_source_video_frames = False

    def default_config(self):
        return AreaQCConfig()

    def config_to_dict(self, config):
        return {"max_relative_deviation": config.max_relative_deviation, "enabled": config.enabled}

    def enabled(self, config) -> bool:
        """Return whether area-deviation quality control is enabled."""
        return config.enabled

    def run(self, detections, config, frames):
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
    summary_defaults = {"area_rejected": 0}

    def summary_values(self, edges):
        """Count frames rejected by this check."""
        return {
            "area_rejected": sum(
                QCFlag.AREA_DEVIATION in detection.qc.flags
                for detection in edges.successful_detections
            )
        }

    diagnostic_writers = (
        ('.area_qc.png', _save_area_qc_plot),
        ('.area_qc.csv', _write_area_qc_csv),
    )
