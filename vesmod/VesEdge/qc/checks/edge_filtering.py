#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quality-control algorithms for VesEdge edge detections."""

from dataclasses import dataclass

import numpy as np

from ..contract import QCCheck
from ...models import EdgeDetection
from ...contour_geometry import measure_wrapped_finite_second_difference
from vesmod.validation import require_nonnegative_real



DEFAULT_CURVATURE_THRESHOLD = 21.520619405632544


CHECK_NAME = "curvature"


@dataclass(frozen=True)
class CurvatureQCConfig:
    """Configuration owned by the curvature QC check."""

    threshold: float = DEFAULT_CURVATURE_THRESHOLD
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", require_nonnegative_real(self.threshold, "threshold"))
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")


@dataclass(frozen=True)
class CurvatureQCResult:
    """Per-trajectory outcome produced by the curvature QC check."""

    scores: tuple[float, ...]
    rejected_count: int


def check_curvature(
    edge: EdgeDetection,
    threshold: float,
) -> None:
    """Check a detected edge for excessive local curvature.

    The edge fails curvature QC when the largest absolute wrapped finite
    second difference of its median-radius-normalized analysis contour exceeds
    ``threshold`` after angular-spacing normalization. The score is invariant
    to uniform spatial scaling and angular resampling of the contour.

    Raises
    ------
    ValueError
        If ``threshold`` is non-finite or negative.
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    normalized_radii = (
        edge.analysis_contour.r / np.median(edge.analysis_contour.r)
    )
    angular_step = 2.0 * np.pi / normalized_radii.size
    finite_second_difference = measure_wrapped_finite_second_difference(
        normalized_radii
    )

    if not np.all(np.isfinite(finite_second_difference)):
        edge.qc.diagnostics["curvature_score"] = np.nan
        edge.qc.flags.add(CHECK_NAME)
        return

    curvature_score = float(
        np.max(np.abs(finite_second_difference)) / angular_step**2
    )
    edge.qc.diagnostics["curvature_score"] = curvature_score

    if curvature_score > threshold:
        edge.qc.flags.add(CHECK_NAME)
    else:
        edge.qc.flags.discard(CHECK_NAME)


class CurvatureCheck(QCCheck):
    """Apply curvature-score quality control to individual detections."""

    name = CHECK_NAME
    config_type = CurvatureQCConfig
    def run(self, detections, config, frames):
        """Record curvature scores and flags for each detection."""
        del frames
        for detection in detections:
            check_curvature(detection, threshold=config.threshold)
        scores = tuple(
            float(detection.qc.diagnostics["curvature_score"])
            if "curvature_score" in detection.qc.diagnostics
            else float("nan")
            for detection in detections
        )
        return CurvatureQCResult(
            scores=scores,
            rejected_count=sum(CHECK_NAME in detection.qc.flags for detection in detections),
        )

    def add_cli_arguments(self, parser) -> None:
        """Register command-line options for this check."""
        parser.add_argument(
            "--curvature-threshold",
            type=float,
            default=DEFAULT_CURVATURE_THRESHOLD,
            help=(
                "Maximum dimensionless angular-resolution-normalized wrapped "
                "finite second difference of the median-radius-normalized "
                "contour allowed by curvature QC. Default: 21.520619405632544."
            ),
        )
        parser.add_argument(
            "--no-curvature-qc",
            action="store_true",
            help="Disable frame-level curvature QC.",
        )

    def config_from_args(self, args):
        """Build settings from parsed command-line options."""
        return CurvatureQCConfig(
            threshold=args.curvature_threshold,
            enabled=not args.no_curvature_qc,
        )

    summary_order = 0
