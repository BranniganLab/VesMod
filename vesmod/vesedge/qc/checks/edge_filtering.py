#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quality-control algorithms for VesEdge edge detections."""

from dataclasses import dataclass
import warnings

import numpy as np

from ..qc_check import QCCheck
from ..qc_result import QCResult
from ...models import EdgeDetection
from ...contour_geometry import measure_wrapped_finite_second_difference
from vesmod.validation import require_nonnegative_real


DEFAULT_CURVATURE_THRESHOLD = 21.520619405632544


CHECK_NAME = "curvature"


@dataclass(frozen=True)
class CurvatureQCConfig:
    """Curvature settings; warn when threshold is exactly the old default 0.059."""

    threshold: float = DEFAULT_CURVATURE_THRESHOLD
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", require_nonnegative_real(self.threshold, "threshold"))
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        if self.threshold == 0.059:
            warnings.warn(
                "Curvature threshold 0.059 appears to be an old default value. "
                "Curvature is now normalized. Consider updating your threshold.",
                UserWarning,
                stacklevel=3,
            )


def check_curvature(
    detections: list[EdgeDetection], threshold: float,
) -> QCResult:
    """Return curvature decisions and dimensionless scores for every contour.

    Reject when the largest absolute wrapped second difference of the
    median-radius-normalized analysis contour, divided by angular spacing
    squared, is greater than threshold. Nonfinite scores also reject.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative.")
    scores = []
    for edge in detections:
        normalized = edge.analysis_contour.r / np.median(edge.analysis_contour.r)
        difference = measure_wrapped_finite_second_difference(normalized)
        angular_step = 2.0 * np.pi / normalized.size
        score = (float(np.max(np.abs(difference)) / angular_step**2)
                 if np.all(np.isfinite(difference)) else float("nan"))
        scores.append(score)
    return QCResult(
        check_name=CHECK_NAME,
        frame_indices=tuple(edge.frame_index for edge in detections),
        rejected=tuple(bool(not np.isfinite(score) or score > threshold) for score in scores),
        diagnostics={"curvature_score": tuple(scores)},
    )


class CurvatureCheck(QCCheck):
    """Apply curvature-score quality control to individual detections."""

    name = CHECK_NAME
    diagnostic_fields = ('curvature_score',)
    config_type = CurvatureQCConfig

    def run(self, detections, config, frames) -> QCResult:
        """Return curvature decisions and scores for each detection."""
        del frames
        return check_curvature(detections, threshold=config.threshold)

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
