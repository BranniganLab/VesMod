"""Tests for detecting a wrongly traced vesicle inside a larger vesicle."""

import numpy as np

from vesmod.VesEdge import EdgeQCConfig
from vesmod.VesEdge.internal_vesicle_qc import (
    check_internal_vesicle_selection,
)
from vesmod.VesEdge.models import EdgeDetection, ImageContour, QCFlag


def _detection(radius: float, frame_index: int = 0) -> EdgeDetection:
    contour = ImageContour(
        (50.0, 50.0),
        np.full(120, radius, dtype=float),
    )
    return EdgeDetection(contour, contour, frame_index=frame_index)


def _ring_frame(*radii: float) -> np.ndarray:
    y, x = np.indices((100, 100))
    radial_distance = np.hypot(x - 50.0, y - 50.0)
    frame = np.zeros((100, 100), dtype=float)
    for radius in radii:
        frame += np.exp(-0.5 * ((radial_distance - radius) / 1.5) ** 2)
    return frame


def test_large_selected_edge_skips_internal_vesicle_inspection():
    """A contour occupying at least half the image cannot be internal."""
    detection = _detection(radius=40.0)
    config = EdgeQCConfig(
        curvature_threshold=1.0,
        enable_internal_vesicle_qc=True,
    )

    result = check_internal_vesicle_selection(
        np.stack([_ring_frame(40.0)]),
        [detection],
        config,
    )

    assert result.inspected is False
    assert result.contour_area_fraction >= 0.5
    assert result.scores == ()
    assert QCFlag.INTERNAL_VESICLE not in detection.qc.flags
    assert detection.qc.internal_vesicle_score is None


def test_persistent_larger_boundary_flags_internal_vesicle_selection():
    """A stable small trace inside a larger membrane rejects the trajectory."""
    detections = [_detection(radius=12.0, frame_index=index) for index in range(4)]
    frames = np.stack([_ring_frame(12.0, 32.0) for _ in detections])
    config = EdgeQCConfig(
        curvature_threshold=1.0,
        enable_internal_vesicle_qc=True,
    )

    result = check_internal_vesicle_selection(frames, detections, config)

    assert result.inspected is True
    assert result.positive_frame_fraction == 1.0
    assert result.rejected_count == len(detections)
    assert all(
        QCFlag.INTERNAL_VESICLE in detection.qc.flags
        for detection in detections
    )


def test_isolated_outer_boundary_does_not_reject_video():
    """Frame aggregation prevents one anomalous frame rejecting a video."""
    detections = [_detection(radius=12.0, frame_index=index) for index in range(4)]
    frames = np.stack(
        [_ring_frame(12.0, 32.0)]
        + [_ring_frame(12.0) for _ in range(3)]
    )
    config = EdgeQCConfig(
        curvature_threshold=1.0,
        enable_internal_vesicle_qc=True,
        internal_vesicle_min_frame_fraction=0.5,
    )

    result = check_internal_vesicle_selection(frames, detections, config)

    assert result.positive_frame_fraction < 0.5
    assert result.rejected_count == 0
    assert all(detection.qc.passed for detection in detections)
