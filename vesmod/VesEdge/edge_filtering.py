#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quality-control algorithms for VesEdge edge detections."""

from dataclasses import dataclass

import numpy as np

from .models import EdgeDetection, QCFlag
from .vesicle_video_utils import measure_wrapped_finite_second_difference
from vesmod.validation import require_nonnegative_real


# Preserve the existing threshold scale while making the score independent
# of the number of angular analysis samples.
CURVATURE_REFERENCE_SAMPLES = 120


@dataclass(frozen=True)
class CurvatureQCConfig:
    """Configuration owned by the curvature QC check."""

    threshold: float
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
    ``threshold`` after angular-spacing normalization. The score uses the
    historical 120-sample threshold convention and is invariant to uniform
    spatial scaling and angular resampling of the contour.

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
    n_samples = normalized_radii.size
    angular_resolution_scale = (
        n_samples / CURVATURE_REFERENCE_SAMPLES
    ) ** 2
    finite_second_difference = measure_wrapped_finite_second_difference(
        normalized_radii
    )

    if not np.all(np.isfinite(finite_second_difference)):
        edge.qc.diagnostics["curvature_score"] = np.nan
        edge.qc.flags.add(QCFlag.CURVATURE)
        return

    curvature_score = float(
        np.max(np.abs(finite_second_difference)) * angular_resolution_scale
    )
    edge.qc.diagnostics["curvature_score"] = curvature_score

    if curvature_score > threshold:
        edge.qc.flags.add(QCFlag.CURVATURE)
    else:
        edge.qc.flags.discard(QCFlag.CURVATURE)
