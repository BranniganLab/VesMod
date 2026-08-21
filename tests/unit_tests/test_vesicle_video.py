"""Unit tests for vesicle_video.py."""

import numpy as np
import pytest

from vesmod.VesEdge import EdgeExtractionConfig, VesicleEdges, VesicleVideo
from vesmod.VesEdge.models import EdgeDetection, EdgeDetectionFailure


@pytest.fixture
def video():
    """Return a VesicleVideo with two blank frames."""
    return VesicleVideo(np.zeros((2, 200, 200)))


@pytest.fixture
def extraction_config():
    """Return standard extraction settings."""
    return EdgeExtractionConfig(
        pixels_per_micron=2.0,
        n_angular_samples=120,
    )


def test_post_init_requires_numpy_array():
    """Test that raw frames must be a NumPy array."""
    with pytest.raises(TypeError):
        VesicleVideo([])


def test_post_init_requires_3d_array():
    """Test that raw frames must be three-dimensional."""
    with pytest.raises(IndexError):
        VesicleVideo(np.zeros((10, 10)))


def test_extract_edges_returns_vesicle_edges(
    video,
    extraction_config,
):
    """Test that extraction returns a separate VesicleEdges object."""
    def extractor(frame):
        return np.full(200, 20.0), (60.0, 50.0)

    edges = video.extract_edges(
        extractor,
        extraction_config,
    )

    assert isinstance(edges, VesicleEdges)
    assert edges.qc_config is None
    assert len(edges.detections) == video.frames.shape[0]
    assert all(
        isinstance(result, EdgeDetection)
        for result in edges.detections
    )


def test_extract_edges_preserves_frame_order_after_failure(
    extraction_config,
):
    """Test that extraction failures retain their original frame position."""
    frames = np.zeros((3, 200, 200))
    frames[1] = 1.0
    video = VesicleVideo(frames)

    def extractor(frame):
        if np.all(frame == 1.0):
            raise RuntimeError("failure")
        return np.full(200, 20.0), (60.0, 50.0)

    edges = video.extract_edges(
        extractor,
        extraction_config,
    )

    assert isinstance(edges.detections[0], EdgeDetection)
    assert isinstance(edges.detections[1], EdgeDetectionFailure)
    assert isinstance(edges.detections[2], EdgeDetection)


def test_extract_edges_records_extractor_index_error(
    extraction_config,
):
    """Test an IndexError raised by an extractor remains a frame failure."""
    frames = np.zeros((2, 200, 200))
    frames[0] = 1.0
    video = VesicleVideo(frames)

    def extractor(frame):
        if np.all(frame == 1.0):
            raise IndexError("extractor index failure")
        return np.full(200, 20.0), (60.0, 50.0)

    edges = video.extract_edges(extractor, extraction_config)

    assert isinstance(edges.detections[0], EdgeDetectionFailure)
    assert edges.detections[0].error == "extractor index failure"
    assert isinstance(edges.detections[1], EdgeDetection)


def test_extract_edges_propagates_invalid_downsampling_configuration(video):
    """Test an impossible analysis sample count is a configuration failure."""
    config = EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=240,
    )

    def extractor(frame):
        return np.full(200, 20.0), (60.0, 50.0)

    with pytest.raises(IndexError, match="Cannot downsample"):
        video.extract_edges(extractor, config)


def test_extract_edges_raises_when_all_extractions_fail(
    video,
    extraction_config,
):
    """Test that extraction reports failure when no frame yields a detection."""
    def failing_extractor(frame):
        raise RuntimeError("custom extractor failed")

    with pytest.raises(
        ValueError,
        match="custom extractor failed",
    ):
        video.extract_edges(
            failing_extractor,
            extraction_config,
        )


def test_extract_edges_rejects_inconsistent_analysis_lengths(video):
    """Test that stored analysis contours must have consistent lengths."""
    config = EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=None,
    )
    video.frames[1] = 1.0

    def extractor(frame):
        if np.all(frame == 1.0):
            return np.full(180, 20.0), (60.0, 50.0)
        return np.full(200, 20.0), (60.0, 50.0)

    with pytest.raises(
        ValueError,
        match="inconsistent numbers of angular samples",
    ):
        video.extract_edges(extractor, config)


def test_extract_edges_does_not_run_qc(video, extraction_config):
    """Test that extraction leaves successful detections without QC state."""
    def extractor(frame):
        radii = np.full(200, 20.0)
        radii[50] = 100.0
        return radii, (60.0, 50.0)

    edges = video.extract_edges(
        extractor,
        extraction_config,
    )

    assert edges.qc_config is None
    assert all(
        detection.qc.curvature_score is None
        for detection in edges.successful_detections
    )


def test_compile_edge_detection_results_preserves_and_downsamples_contour(
    extraction_config,
):
    """Test contour compilation preserves native data and downsampling."""
    r_vals = np.linspace(10.0, 20.0, 200)
    edge = VesicleVideo._compile_edge_detection_results(
        r_vals,
        (60.0, 50.0),
        extraction_config,
    )

    assert edge.full_contour.origin == (50.0, 60.0)
    assert edge.full_contour.r.shape == (200,)
    assert edge.analysis_contour.r.shape == (120,)


def test_downsample_r_vals_rejects_more_samples_than_input():
    """Test that extraction cannot upsample through the downsampling path."""
    with pytest.raises(IndexError, match="Cannot downsample"):
        VesicleVideo._downsample_r_vals(
            np.ones(10),
            20,
        )


def test_make_vesicle_gif_rejects_mismatched_edges(
    tmp_path,
    extraction_config,
):
    """Test that an overlay must correspond to every video frame."""
    video = VesicleVideo(np.zeros((2, 20, 20)))
    edge = VesicleVideo._compile_edge_detection_results(
        np.full(20, 5.0),
        (10.0, 10.0),
        EdgeExtractionConfig(
            pixels_per_micron=1.0,
            n_angular_samples=20,
        ),
    )
    edges = VesicleEdges(
        extraction_config=extraction_config,
        detections=[edge],
    )

    with pytest.raises(ValueError, match="1 detections and 2 frames"):
        video.make_vesicle_gif(
            tmp_path / "video.gif",
            edges,
        )
