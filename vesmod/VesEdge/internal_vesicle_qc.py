"""Spatial QC for persistent selection of a vesicle inside another vesicle."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, map_coordinates

from .area_qc import contour_area
from .config import EdgeQCConfig
from .models import EdgeDetection, InternalVesicleQCResult, QCFlag


def _frame_enclosing_boundary_score(
    frame: NDArray[np.number],
    detection: EdgeDetection,
    config: EdgeQCConfig,
) -> float:
    """Return angular coverage by strong gradients beyond the selected edge."""
    image = np.asarray(frame, dtype=float)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        return float("nan")

    contour = detection.full_contour
    theta = contour.theta
    height, width = image.shape
    max_radius = float(np.hypot(height, width))
    radii = np.arange(0.0, max_radius + 1.0)
    sample_x = contour.origin[0] + np.cos(theta)[:, None] * radii
    sample_y = contour.origin[1] + np.sin(theta)[:, None] * radii
    inside = (
        (sample_x >= 0)
        & (sample_x <= width - 1)
        & (sample_y >= 0)
        & (sample_y <= height - 1)
    )
    profiles = map_coordinates(
        image,
        [sample_y.ravel(), sample_x.ravel()],
        order=1,
        mode="nearest",
    ).reshape(sample_x.shape)
    gradients = np.abs(gaussian_filter1d(profiles, sigma=1.0, axis=1, order=1))

    selected = contour.r
    selected_strengths = np.empty(theta.size, dtype=float)
    outer_strengths = np.full(theta.size, np.nan, dtype=float)
    for index, selected_radius in enumerate(selected):
        edge_start = max(0, int(np.floor(selected_radius)) - 2)
        edge_stop = min(gradients.shape[1], int(np.ceil(selected_radius)) + 3)
        selected_strengths[index] = np.max(gradients[index, edge_start:edge_stop])

        outer_start = int(
            np.ceil(
                max(
                    selected_radius
                    * config.internal_vesicle_min_radius_ratio,
                    selected_radius
                    + config.internal_vesicle_min_separation_pixels,
                )
            )
        )
        valid_indices = np.flatnonzero(inside[index] & (radii >= outer_start))
        if valid_indices.size:
            outer_strengths[index] = np.max(gradients[index, valid_indices])

    finite_selected = selected_strengths[np.isfinite(selected_strengths)]
    reference_strength = (
        float(np.median(finite_selected))
        if finite_selected.size
        else float("nan")
    )
    if not np.isfinite(reference_strength) or reference_strength <= 0:
        return float("nan")
    strong_outer_edge = outer_strengths >= (
        config.internal_vesicle_gradient_ratio * reference_strength
    )
    usable = np.isfinite(outer_strengths)
    return (
        float(np.mean(strong_outer_edge[usable]))
        if np.any(usable)
        else float("nan")
    )


def check_internal_vesicle_selection(
    frames: NDArray[np.number],
    detections: list[EdgeDetection],
    config: EdgeQCConfig,
) -> InternalVesicleQCResult:
    """Flag a trajectory when a larger enclosing boundary persists over time."""
    video = np.asarray(frames)
    if video.ndim != 3:
        raise ValueError("Internal-vesicle QC requires a 3D frame array.")
    if not detections:
        raise ValueError("Internal-vesicle QC requires successful detections.")
    if any(
        edge.frame_index is None or edge.frame_index >= video.shape[0]
        for edge in detections
    ):
        raise ValueError(
            "Internal-vesicle QC frames do not match detection indices."
        )

    median_area = float(
        np.median(
            [contour_area(edge.full_contour.r) for edge in detections]
        )
    )
    frame_area = float(video.shape[1] * video.shape[2])
    area_fraction = median_area / frame_area
    if area_fraction >= config.max_internal_vesicle_area_fraction:
        return InternalVesicleQCResult(
            inspected=False,
            contour_area_fraction=area_fraction,
            scores=(),
            positive_frame_fraction=0.0,
            rejected_count=0,
            reason=(
                "Selected contour occupies too much of the frame to plausibly "
                "be an internal vesicle; enclosing-boundary inspection skipped."
            ),
        )

    scores = tuple(
        _frame_enclosing_boundary_score(video[edge.frame_index], edge, config)
        for edge in detections
    )
    for edge, score in zip(detections, scores, strict=True):
        edge.qc.internal_vesicle_score = score
    finite_scores = np.asarray(scores)[np.isfinite(scores)]
    positive_fraction = (
        float(
            np.mean(
                finite_scores
                >= config.internal_vesicle_min_angular_coverage
            )
        )
        if finite_scores.size
        else 0.0
    )
    reject = (
        finite_scores.size > 0
        and positive_fraction >= config.internal_vesicle_min_frame_fraction
    )
    if reject:
        for edge in detections:
            edge.qc.flags.add(QCFlag.INTERNAL_VESICLE)
    return InternalVesicleQCResult(
        inspected=True,
        contour_area_fraction=area_fraction,
        scores=scores,
        positive_frame_fraction=positive_fraction,
        rejected_count=len(detections) if reject else 0,
        reason=(
            "Persistent larger enclosing boundary detected."
            if reject
            else "No persistent larger enclosing boundary detected."
        ),
    )
