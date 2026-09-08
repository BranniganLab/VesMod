"""Experimental QC for selecting a vesicle inside another vesicle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter

from ..area_qc import contour_area
from ..models import EdgeDetection
from vesmod.validation import (
    require_fraction,
    require_integer_valued,
    require_nonnegative_real,
    require_positive_real,
)

if TYPE_CHECKING:
    from ..frame_source import FrameSource


@dataclass(frozen=True)
class InternalVesicleQCConfig:
    """Configuration owned by the internal-vesicle QC check."""

    enabled: bool = False
    max_area_fraction: float = 0.5
    min_radius_ratio: float = 1.15
    min_separation_fraction: float = 0.4
    gradient_ratio: float = 0.5
    max_radial_deviation_fraction: float = 0.15
    min_angular_coverage: float = 0.6
    max_frames: int = 20
    min_valid_frames: int = 3
    min_valid_frame_fraction: float = 0.5
    min_frame_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        for name in (
            "max_area_fraction", "max_radial_deviation_fraction",
            "min_angular_coverage", "min_valid_frame_fraction", "min_frame_fraction",
        ):
            object.__setattr__(self, name, require_fraction(getattr(self, name), name))
        minimum_ratio = require_positive_real(self.min_radius_ratio, "min_radius_ratio")
        if minimum_ratio <= 1:
            raise ValueError("min_radius_ratio must be greater than 1.")
        object.__setattr__(self, "min_radius_ratio", minimum_ratio)
        object.__setattr__(self, "min_separation_fraction", require_fraction(self.min_separation_fraction, "min_separation_fraction"))
        object.__setattr__(self, "gradient_ratio", require_nonnegative_real(self.gradient_ratio, "gradient_ratio"))
        for name in ("max_frames", "min_valid_frames"):
            value = require_integer_valued(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class InternalVesicleQCResult:
    """Per-trajectory outcome produced by internal-vesicle QC."""

    inspected: bool
    contour_area_fraction: float
    sampled_frame_indices: tuple[int, ...]
    scores: tuple[float, ...]
    valid_frame_count: int
    valid_frame_fraction: float
    positive_frame_fraction: float
    persistent_enclosing_boundary: bool
    reason: str


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
    config: InternalVesicleQCConfig,
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
    allowed_deviation = config.max_radial_deviation_fraction * smooth_radii
    coherent = np.abs(outer_radii - smooth_radii) <= allowed_deviation
    strong = outer_strengths >= (
        config.gradient_ratio * reference_strength
    )
    return float(np.count_nonzero(coherent & strong & usable) / outer_radii.size)


def _frame_enclosing_boundary_score(
    frame: NDArray[np.number],
    detection: EdgeDetection,
    config: InternalVesicleQCConfig,
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
                    * config.min_radius_ratio,
                    selected_radius * (1 + config.min_separation_fraction),
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
    config: InternalVesicleQCConfig,
) -> InternalVesicleQCResult:
    """Evaluate persistent selection of a smaller vesicle within a larger one."""
    from ..frame_source import as_frame_source

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
    internal_config = config
    if area_fraction >= internal_config.max_area_fraction:
        return InternalVesicleQCResult(
            inspected=False,
            contour_area_fraction=area_fraction,
            sampled_frame_indices=(),
            scores=(),
            valid_frame_count=0,
            valid_frame_fraction=0.0,
            positive_frame_fraction=0.0,
            persistent_enclosing_boundary=False,
            reason=(
                "Selected contour occupies too much of the frame to plausibly "
                "be an internal vesicle; enclosing-boundary inspection skipped."
            ),
        )

    sampled = _sample_detections(
        detections,
        internal_config.max_frames,
    )
    sampled_indices = tuple(edge.frame_index for edge in sampled)
    scores = tuple(
        _frame_enclosing_boundary_score(
            frame_source[edge.frame_index],
            edge,
            internal_config,
        )
        for edge in sampled
    )
    for edge, score in zip(sampled, scores, strict=True):
        edge.qc.diagnostics["internal_vesicle_score"] = score

    finite_scores = np.asarray(scores)[np.isfinite(scores)]
    valid_count = int(finite_scores.size)
    valid_fraction = valid_count / len(sampled)
    positive_fraction = (
        float(
            np.mean(
                finite_scores
                >= internal_config.min_angular_coverage
            )
        )
        if valid_count
        else 0.0
    )
    required_valid_count = min(
        internal_config.min_valid_frames,
        len(sampled),
    )
    sufficient_valid_data = (
        valid_count >= required_valid_count
        and valid_fraction >= internal_config.min_valid_frame_fraction
    )
    reject = (
        sufficient_valid_data
        and positive_fraction >= internal_config.min_frame_fraction
    )
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
        persistent_enclosing_boundary=reject,
        reason=reason,
    )
