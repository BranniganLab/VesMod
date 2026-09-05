"""Experimental QC for selecting a vesicle inside another vesicle."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter

from ..area_qc import contour_area
from ..config import EdgeQCConfig
from ..frame_source import FrameSource, as_frame_source
from ..models import EdgeDetection, InternalVesicleQCResult, QCFlag


def _sample_detections(
    detections: Sequence[EdgeDetection],
    max_frames: int,
) -> list[EdgeDetection]:
    """Select detections evenly across the trajectory in temporal order."""
    if len(detections) <= max_frames:
        return list(detections)
    positions = np.linspace(
        0,
        len(detections) - 1,
        num=max_frames,
        dtype=int,
    )
    return [detections[position] for position in positions]


def _coherent_outer_edge_coverage(
    outer_radii: NDArray[np.float64],
    outer_strengths: NDArray[np.float64],
    reference_strength: float,
    config: EdgeQCConfig,
) -> float:
    """Return coverage by strong peaks belonging to one smooth outer contour."""
    usable = np.isfinite(outer_radii) & np.isfinite(outer_strengths)
    if not np.any(usable):
        return float("nan")

    positions = np.arange(outer_radii.size)
    usable_positions = positions[usable]
    usable_radii = outer_radii[usable]
    filled_radii = np.interp(
        positions,
        np.concatenate(
            (
                usable_positions - outer_radii.size,
                usable_positions,
                usable_positions + outer_radii.size,
            )
        ),
        np.tile(usable_radii, 3),
    )
    smoothing_width = min(11, outer_radii.size)
    if smoothing_width % 2 == 0:
        smoothing_width -= 1
    smooth_radii = median_filter(
        filled_radii,
        size=max(1, smoothing_width),
        mode="wrap",
    )
    allowed_deviation = np.maximum(
        config.internal_vesicle_min_separation_pixels,
        config.internal_vesicle_max_radial_deviation_fraction * smooth_radii,
    )
    coherent = np.abs(outer_radii - smooth_radii) <= allowed_deviation
    strong = outer_strengths >= (
        config.internal_vesicle_gradient_ratio * reference_strength
    )
    return float(np.mean((coherent & strong)[usable]))


def _frame_enclosing_boundary_score(
    frame: NDArray[np.number],
    detection: EdgeDetection,
    config: EdgeQCConfig,
) -> float:
    """Return coherent angular coverage by gradients beyond the selected edge."""
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

    selected_strengths = np.empty(theta.size, dtype=float)
    outer_strengths = np.full(theta.size, np.nan, dtype=float)
    outer_radii = np.full(theta.size, np.nan, dtype=float)
    for index, selected_radius in enumerate(contour.r):
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
            strongest = valid_indices[
                np.argmax(gradients[index, valid_indices])
            ]
            outer_strengths[index] = gradients[index, strongest]
            outer_radii[index] = radii[strongest]

    finite_selected = selected_strengths[np.isfinite(selected_strengths)]
    reference_strength = (
        float(np.median(finite_selected))
        if finite_selected.size
        else float("nan")
    )
    if not np.isfinite(reference_strength) or reference_strength <= 0:
        return float("nan")
    return _coherent_outer_edge_coverage(
        outer_radii,
        outer_strengths,
        reference_strength,
        config,
    )


def check_internal_vesicle_selection(
    frames: FrameSource | NDArray[np.number],
    detections: list[EdgeDetection],
    config: EdgeQCConfig,
) -> InternalVesicleQCResult:
    """Flag persistent selection of a smaller vesicle within a larger one."""
    frame_source = as_frame_source(frames)
    frame_count, height, width = frame_source.shape
    if not detections:
        raise ValueError("Internal-vesicle QC requires successful detections.")
    if any(
        edge.frame_index is None
        or edge.frame_index < 0
        or edge.frame_index >= frame_count
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
    frame_area = float(height * width)
    area_fraction = median_area / frame_area
    if area_fraction >= config.max_internal_vesicle_area_fraction:
        return InternalVesicleQCResult(
            inspected=False,
            contour_area_fraction=area_fraction,
            sampled_frame_indices=(),
            scores=(),
            valid_frame_count=0,
            valid_frame_fraction=0.0,
            positive_frame_fraction=0.0,
            rejected_count=0,
            reason=(
                "Selected contour occupies too much of the frame to plausibly "
                "be an internal vesicle; enclosing-boundary inspection skipped."
            ),
        )

    sampled = _sample_detections(
        detections,
        config.internal_vesicle_max_frames,
    )
    sampled_indices = tuple(edge.frame_index for edge in sampled)
    scores = tuple(
        _frame_enclosing_boundary_score(
            frame_source[edge.frame_index],
            edge,
            config,
        )
        for edge in sampled
    )
    for edge, score in zip(sampled, scores, strict=True):
        edge.qc.internal_vesicle_score = score

    finite_scores = np.asarray(scores)[np.isfinite(scores)]
    valid_count = int(finite_scores.size)
    valid_fraction = valid_count / len(sampled)
    positive_fraction = (
        float(
            np.mean(
                finite_scores
                >= config.internal_vesicle_min_angular_coverage
            )
        )
        if valid_count
        else 0.0
    )
    required_valid_count = min(
        config.internal_vesicle_min_valid_frames,
        len(sampled),
    )
    sufficient_valid_data = (
        valid_count >= required_valid_count
        and valid_fraction >= config.internal_vesicle_min_valid_frame_fraction
    )
    reject = (
        sufficient_valid_data
        and positive_fraction >= config.internal_vesicle_min_frame_fraction
    )
    if reject:
        for edge in detections:
            edge.qc.flags.add(QCFlag.INTERNAL_VESICLE)

    if not sufficient_valid_data:
        reason = "Insufficient valid sampled frames for internal-vesicle QC."
    elif reject:
        reason = "Persistent larger enclosing boundary detected."
    else:
        reason = "No persistent larger enclosing boundary detected."
    return InternalVesicleQCResult(
        inspected=True,
        contour_area_fraction=area_fraction,
        sampled_frame_indices=sampled_indices,
        scores=scores,
        valid_frame_count=valid_count,
        valid_frame_fraction=valid_fraction,
        positive_frame_fraction=positive_fraction,
        rejected_count=len(detections) if reject else 0,
        reason=reason,
    )
