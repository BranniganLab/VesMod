"""Tests for detecting a wrongly traced vesicle inside a larger vesicle."""

import numpy as np
import pytest

from vesmod.VesEdge import EdgeQCConfig
from vesmod.VesEdge.experimental.internal_vesicle_qc import (
    _coherent_outer_edge_coverage,
    _frame_enclosing_boundary_score,
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
    assert result.sampled_frame_indices == ()
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


def test_incoherent_outer_peaks_do_not_form_enclosing_boundary():
    """Strong gradients at unrelated radii do not count as one membrane."""
    rng = np.random.default_rng(1234)
    outer_radii = rng.uniform(20.0, 60.0, size=120)
    outer_strengths = np.ones(120)
    config = EdgeQCConfig(
        curvature_threshold=1.0,
        enable_internal_vesicle_qc=True,
    )

    score = _coherent_outer_edge_coverage(
        outer_radii,
        outer_strengths,
        reference_strength=1.0,
        config=config.internal_vesicle,
    )

    assert score < config.internal_vesicle.min_angular_coverage


def test_clipped_directions_count_as_missing_outer_boundary_evidence():
    """A partial ring at an image border does not imply full coverage."""
    y, x = np.indices((100, 100))
    distance = np.hypot(x, y - 50.0)
    frame = sum(
        np.exp(-0.5 * ((distance - radius) / 1.5) ** 2)
        for radius in (12.0, 32.0)
    )
    contour = ImageContour((0.0, 50.0), np.full(120, 12.0))
    detection = EdgeDetection(contour, contour, frame_index=0)
    config = EdgeQCConfig(
        curvature_threshold=1.0,
        enable_internal_vesicle_qc=True,
    )

    score = _frame_enclosing_boundary_score(
        frame, detection, config.internal_vesicle
    )

    assert np.isfinite(score)
    assert score < config.internal_vesicle.min_angular_coverage


def test_size_gate_does_not_read_lazy_frames():
    """Large contours are dismissed using metadata before any frame read."""
    source = _CountingFrameSource(np.stack([_ring_frame(40.0)] * 5))
    detections = [_detection(40.0, index) for index in range(5)]
    config = EdgeQCConfig(
        curvature_threshold=1.0,
        enable_internal_vesicle_qc=True,
    )

    result = check_internal_vesicle_selection(source, detections, config)

    assert result.inspected is False
    assert source.read_indices == []


def test_sampling_reads_only_evenly_spaced_frames():
    """Inspection stays bounded by the configured frame sample."""
    source = _CountingFrameSource(np.stack([_ring_frame(12.0)] * 10))
    detections = [_detection(12.0, index) for index in range(10)]
    config = EdgeQCConfig(
        curvature_threshold=1.0,
        enable_internal_vesicle_qc=True,
        internal_vesicle_max_frames=4,
    )

    result = check_internal_vesicle_selection(source, detections, config)

    assert result.sampled_frame_indices == (0, 3, 6, 9)
    assert source.read_indices == [0, 3, 6, 9]


def test_insufficient_valid_sample_cannot_reject_trajectory():
    """One usable frame cannot decide a trajectory when coverage is poor."""
    frames = np.stack(
        [_ring_frame(12.0, 32.0)]
        + [np.full((100, 100), np.nan) for _ in range(3)]
    )
    detections = [_detection(12.0, index) for index in range(4)]
    config = EdgeQCConfig(
        curvature_threshold=1.0,
        enable_internal_vesicle_qc=True,
        internal_vesicle_min_valid_frames=3,
        internal_vesicle_min_valid_frame_fraction=0.5,
    )

    result = check_internal_vesicle_selection(frames, detections, config)

    assert result.valid_frame_count == 1
    assert result.valid_frame_fraction == 0.25
    assert result.rejected_count == 0
    assert result.reason.startswith("Insufficient")


def test_negative_frame_index_is_rejected():
    """Negative indices must not silently select frames from the video end."""
    detection = _detection(radius=12.0, frame_index=-1)
    config = EdgeQCConfig(
        curvature_threshold=1.0,
        enable_internal_vesicle_qc=True,
    )

    with pytest.raises(ValueError, match="do not match detection indices"):
        check_internal_vesicle_selection(
            np.stack([_ring_frame(12.0, 32.0)]),
            [detection],
            config,
        )


class _CountingFrameSource:
    """Minimal lazy source that records indexed reads."""

    def __init__(self, frames):
        self._frames = frames
        self.read_indices = []

    @property
    def shape(self):
        return self._frames.shape

    @property
    def metadata(self):
        return {"kind": "test"}

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        self.read_indices.append(index)
        return self._frames[index]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
