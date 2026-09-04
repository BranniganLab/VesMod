#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quality-control algorithms for VesEdge edge detections."""

import numpy as np

from .models import EdgeDetection, QCFlag
from .vesicle_video_utils import measure_wrapped_finite_second_difference


def check_curvature(
    edge: EdgeDetection,
    threshold: float,
) -> None:
    """Check a detected edge for excessive local curvature.

    The edge fails curvature QC when the largest absolute wrapped finite
    second difference of its median-radius-normalized analysis contour exceeds
    ``threshold``. The resulting dimensionless score is invariant to uniform
    spatial scaling of the contour.

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
    finite_second_difference = measure_wrapped_finite_second_difference(
        normalized_radii
    )

    if not np.all(np.isfinite(finite_second_difference)):
        edge.qc.curvature_score = np.nan
        edge.qc.flags.add(QCFlag.CURVATURE)
        return

    curvature_score = float(
        np.max(np.abs(finite_second_difference))
    )
    edge.qc.curvature_score = curvature_score

    if curvature_score > threshold:
        edge.qc.flags.add(QCFlag.CURVATURE)
    else:
        edge.qc.flags.discard(QCFlag.CURVATURE)
