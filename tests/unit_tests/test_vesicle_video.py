"""Unit tests for vesicle_video.py."""
import numpy as np
import pytest

from vesmod.VesEdge.edge_filtering import EdgeQCConfig
from vesmod.VesEdge.models import (
    EdgeDetection,
    EdgeDetectionFailure,
    QCFlag,
)
from vesmod.VesEdge.vesicle_video import (
    EdgeExtractionConfig,
    VesicleVideo,
)


@pytest.fixture
def extraction_config():
    """Return a standard edge-extraction configuration for tests."""
    return EdgeExtractionConfig(
        pixels_per_micron=2.0,
        n_angular_samples=120,
    )


@pytest.fixture
def qc_config():
    """Return a standard edge-QC configuration for tests."""
    return EdgeQCConfig(
        curvature_threshold=5.0,
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
    )


@pytest.fixture
def video(extraction_config, qc_config):
    """Return a VesicleVideo with one blank frame."""
    return VesicleVideo(
        np.zeros((1, 200, 200)),
        extraction_config,
        qc_config,
    )


def test_post_init_starts_with_empty_detections(
    extraction_config,
    qc_config,
):
    """Test that a newly constructed video has no edge-extraction results."""
    video = VesicleVideo(
        np.zeros((3, 200, 200)),
        extraction_config,
        qc_config,
    )

    assert video.detections == []


def test_post_init_requires_numpy_array(
    extraction_config,
    qc_config,
):
    """Test that frames must be a NumPy ndarray."""
    with pytest.raises(TypeError):
        VesicleVideo(
            [],
            extraction_config,
            qc_config,
        )


def test_post_init_requires_3d_array(
    extraction_config,
    qc_config,
):
    """Test that frames must be three-dimensional."""
    with pytest.raises(IndexError):
        VesicleVideo(
            np.zeros((10, 10)),
            extraction_config,
            qc_config,
        )


def test_edge_extraction_config_requires_positive_pixels_per_micron():
    """Test that pixels_per_micron must be positive."""
    with pytest.raises(ValueError):
        EdgeExtractionConfig(
            pixels_per_micron=0,
            n_angular_samples=120,
        )


def test_edge_extraction_config_converts_integer_valued_sample_count():
    """Test that integer-valued sample counts are normalized to int."""
    config = EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=120.0,
    )

    assert config.n_angular_samples == 120
    assert isinstance(
        config.n_angular_samples,
        int,
    )


def test_edge_extraction_config_rejects_non_integer_sample_count():
    """Test that non-integer sample counts are rejected."""
    with pytest.raises(
        ValueError,
        match="integer-valued",
    ):
        EdgeExtractionConfig(
            pixels_per_micron=1.0,
            n_angular_samples=120.5,
        )


def test_edge_extraction_config_requires_positive_n_angular_samples():
    """Test that n_angular_samples must be positive when supplied."""
    with pytest.raises(ValueError):
        EdgeExtractionConfig(
            pixels_per_micron=1.0,
            n_angular_samples=0,
        )


def test_edge_extraction_config_allows_no_downsampling():
    """Test that None disables contour downsampling."""
    config = EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=None,
    )

    assert config.n_angular_samples is None


def test_post_init_rejects_too_many_samples(qc_config):
    """Test that the requested sample count cannot exceed the native contour length."""
    extraction_config = EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=1000,
    )

    with pytest.raises(IndexError):
        VesicleVideo(
            np.zeros((2, 50, 50)),
            extraction_config,
            qc_config,
        )


def test_post_init_allows_no_downsampling(qc_config):
    """Test that a video may be constructed with downsampling disabled."""
    extraction_config = EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=None,
    )

    video = VesicleVideo(
        np.zeros((2, 50, 50)),
        extraction_config,
        qc_config,
    )

    assert video.extraction_config.n_angular_samples is None


def test_downsample_r_vals_returns_input_when_same_length(video):
    """Test that no interpolation occurs when requested and native lengths match."""
    r_vals = np.arange(20, dtype=float)

    returned = video._downsample_r_vals(r_vals, 20)

    assert returned is r_vals


def test_compile_edge_detection_results_stores_full_contour(video):
    """Test that edge compilation preserves the native extracted contour."""
    r_vals = np.full(200, 20.0)
    vesicle_center = (60.0, 50.0)

    edge = video._compile_edge_detection_results(
        r_vals,
        vesicle_center,
    )

    assert isinstance(edge, EdgeDetection)
    assert edge.full_contour.origin == (50.0, 60.0)
    np.testing.assert_array_equal(
        edge.full_contour.r,
        r_vals,
    )


def test_compile_edge_detection_results_downsamples_analysis_contour(video):
    """Test that the analysis contour uses the configured angular sample count."""
    r_vals = np.linspace(10.0, 20.0, 200)

    edge = video._compile_edge_detection_results(
        r_vals,
        (60.0, 50.0),
    )

    assert edge.analysis_contour.r.shape == (120,)
    assert edge.full_contour.r.shape == (200,)


def test_compile_edge_detection_results_converts_radii_to_microns(video):
    """Test that analysis radii are converted from pixels to microns."""
    r_vals = np.full(200, 20.0)

    edge = video._compile_edge_detection_results(
        r_vals,
        (60.0, 50.0),
    )

    np.testing.assert_allclose(
        edge.radii_microns,
        10.0,
    )


def test_compile_edge_detection_results_uses_full_contour_when_not_downsampling(
    qc_config,
):
    """Test that the full contour is also the analysis contour when downsampling is off."""
    extraction_config = EdgeExtractionConfig(
        pixels_per_micron=2.0,
        n_angular_samples=None,
    )
    video = VesicleVideo(
        np.zeros((1, 200, 200)),
        extraction_config,
        qc_config,
    )
    r_vals = np.full(200, 20.0)

    edge = video._compile_edge_detection_results(
        r_vals,
        (60.0, 50.0),
    )

    assert edge.analysis_contour is edge.full_contour
    np.testing.assert_array_equal(
        edge.radii_microns,
        np.full(200, 10.0),
    )


def test_validate_extractor_results_requires_ndarray(video):
    """Test that extractor radii must be returned as a NumPy ndarray."""
    with pytest.raises(TypeError, match="NDArray"):
        video._validate_extractor_results([1.0, 2.0])


def test_validate_extractor_results_requires_one_dimension(video):
    """Test that extractor radii must be one-dimensional."""
    with pytest.raises(ValueError, match="1D"):
        video._validate_extractor_results(np.zeros((2, 2)))


def test_run_frame_qc_calls_all_frame_level_checks(monkeypatch, video):
    """Test that frame QC invokes each configured frame-level QC function."""
    calls = []

    def fake_curvature(edge, threshold):
        calls.append(("curvature", threshold))

    monkeypatch.setattr(
        "vesmod.VesEdge.vesicle_video.check_curvature",
        fake_curvature,
    )

    edge = video._compile_edge_detection_results(
        np.full(200, 20.0),
        (60.0, 50.0),
    )

    video._run_frame_qc(video.frames[0], edge)

    assert calls == [
        ("curvature", video.qc_config.curvature_threshold),
    ]


def test_run_trajectory_qc_calls_population_check(monkeypatch, video):
    """Test that trajectory QC passes detections and configured thresholds."""
    observed = {}

    def fake_population_check(detections, bic_threshold, max_minor_fraction):
        observed["detections"] = detections
        observed["bic_threshold"] = bic_threshold
        observed["max_minor_fraction"] = max_minor_fraction

    monkeypatch.setattr(
        "vesmod.VesEdge.vesicle_video.check_edge_populations",
        fake_population_check,
    )

    video._run_trajectory_qc()

    assert observed["detections"] is video.detections
    assert observed["bic_threshold"] == video.qc_config.population_bic_threshold
    assert (
        observed["max_minor_fraction"] == video.qc_config.max_minor_population_fraction
    )


def test_extract_edges_records_successful_detections(
    monkeypatch,
    extraction_config,
    qc_config,
):
    """Test that successful extractor results are stored as EdgeDetection objects."""
    video = VesicleVideo(
        np.zeros((2, 200, 200)),
        extraction_config,
        qc_config,
    )

    def extractor(frame):
        return np.full(200, 20.0), (60.0, 50.0)

    monkeypatch.setattr(video, "_run_frame_qc", lambda frame, edge: None)
    monkeypatch.setattr(video, "_run_trajectory_qc", lambda: None)

    video.extract_edges(extractor)

    assert len(video.detections) == 2
    assert all(
        isinstance(result, EdgeDetection)
        for result in video.detections
    )


def test_extract_edges_raises_when_all_extractions_fail(
    extraction_config,
    qc_config,
):
    """Test that extraction fails when the extractor fails on every frame."""
    video = VesicleVideo(
        np.zeros((2, 200, 200)),
        extraction_config,
        qc_config,
    )

    def failing_extractor(frame):
        raise RuntimeError("failure")

    with pytest.raises(
        ValueError,
        match="no successful detections",
    ):
        video.extract_edges(failing_extractor)

    assert len(video.detections) == 2
    assert all(
        isinstance(result, EdgeDetectionFailure)
        for result in video.detections
    )


def test_run_frame_qc_skips_curvature_when_disabled(monkeypatch, extraction_config):
    """Test that disabling curvature QC skips the curvature check."""
    qc_config = EdgeQCConfig(
        curvature_threshold=5.0,
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        enable_curvature_qc=False,
    )
    video = VesicleVideo(np.zeros((1, 200, 200)), extraction_config, qc_config)
    calls = []

    monkeypatch.setattr(
        "vesmod.VesEdge.vesicle_video.check_curvature",
        lambda edge, threshold: calls.append(threshold),
    )

    edge = video._compile_edge_detection_results(np.full(200, 20.0), (60.0, 50.0))
    video._run_frame_qc(video.frames[0], edge)

    assert calls == []


def test_extract_edges_preserves_frame_order_after_failure(
    monkeypatch,
    extraction_config,
    qc_config,
):
    """Test that each extraction result retains its position in video frame order."""
    frames = np.zeros((3, 200, 200))
    frames[1] = 1.0

    video = VesicleVideo(
        frames,
        extraction_config,
        qc_config,
    )

    def extractor(frame):
        if np.all(frame == 1.0):
            raise RuntimeError("failure")
        return np.full(200, 20.0), (60.0, 50.0)

    monkeypatch.setattr(video, "_run_frame_qc", lambda frame, edge: None)
    monkeypatch.setattr(video, "_run_trajectory_qc", lambda: None)

    video.extract_edges(extractor)

    assert isinstance(video.detections[0], EdgeDetection)
    assert isinstance(video.detections[1], EdgeDetectionFailure)
    assert isinstance(video.detections[2], EdgeDetection)


def test_extract_edges_rejects_inconsistent_detection_lengths(
        monkeypatch,
        qc_config,
    ):
        """Test that successful detections must have consistent sample counts."""
        extraction_config = EdgeExtractionConfig(
            pixels_per_micron=1.0,
            n_angular_samples=None,
        )
        frames = np.zeros((2, 200, 200))
        frames[1] = 1.0
    
        video = VesicleVideo(
            frames,
            extraction_config,
            qc_config,
        )
    
        def extractor(frame):
            if np.all(frame == 1.0):
                return np.full(180, 20.0), (60.0, 50.0)
            return np.full(200, 20.0), (60.0, 50.0)
    
        monkeypatch.setattr(
            video,
            "_run_frame_qc",
            lambda frame, edge: None,
        )
    
        with pytest.raises(
            ValueError,
            match="inconsistent numbers of angular samples",
        ):
            video.extract_edges(extractor)


def test_extract_edges_raises_when_no_frames_pass_qc(
        monkeypatch,
        extraction_config,
        qc_config,
    ):
        """Test that extraction reports when every detected edge fails QC."""
        video = VesicleVideo(
            np.zeros((2, 200, 200)),
            extraction_config,
            qc_config,
        )
    
        def extractor(frame):
            return np.full(200, 20.0), (60.0, 50.0)
    
        def reject_edge(frame, edge):
            edge.qc.flags.add(QCFlag.CURVATURE)
    
        monkeypatch.setattr(
            video,
            "_run_frame_qc",
            reject_edge,
        )
        monkeypatch.setattr(
            video,
            "_run_trajectory_qc",
            lambda: None,
        )
    
        with pytest.raises(
            ValueError,
            match="no frames passed quality control",
        ):
            video.extract_edges(extractor)


def test_save_edge_to_npy_only_saves_accepted_detections(tmp_path, video):
    """Test that only successfully detected edges that pass QC are saved."""
    accepted_edge = video._compile_edge_detection_results(
        np.full(200, 20.0),
        (60.0, 50.0),
    )
    rejected_edge = video._compile_edge_detection_results(
        np.full(200, 30.0),
        (60.0, 50.0),
    )
    rejected_edge.qc.flags.add(QCFlag.CURVATURE)

    video.detections.extend(
        [
            accepted_edge,
            EdgeDetectionFailure("failure"),
            rejected_edge,
        ]
    )

    outfile = tmp_path / "edges"

    video.save_edge_to_npy(outfile)

    saved = np.load(outfile.with_suffix(".npy"))

    assert saved.shape == (1, 120)
    np.testing.assert_array_equal(
        saved[0],
        accepted_edge.radii_microns,
    )


def test_extract_edges_replaces_previous_detections(
    monkeypatch,
    extraction_config,
    qc_config,
):
    """Test that repeated extraction replaces previous detection results."""
    video = VesicleVideo(
        np.zeros((2, 200, 200)),
        extraction_config,
        qc_config,
    )

    def extractor(frame):
        return np.full(200, 20.0), (60.0, 50.0)

    monkeypatch.setattr(
        video,
        "_run_frame_qc",
        lambda frame, edge: None,
    )
    monkeypatch.setattr(
        video,
        "_run_trajectory_qc",
        lambda: None,
    )

    video.extract_edges(extractor)
    first_detections = list(video.detections)

    video.extract_edges(extractor)

    assert len(video.detections) == video.frames.shape[0]
    assert all(
        second is not first
        for second, first in zip(
            video.detections,
            first_detections,
            strict=True,
        )
    )


def test_make_vesicle_gif_with_trace_requires_detections(
    video,
    tmp_path,
):
    """Verify traced GIFs cannot be rendered before edge extraction."""
    with pytest.raises(
        ValueError,
        match="detections",
    ):
        video.make_vesicle_gif(
            tmp_path / "vesicle.gif",
            show_trace=True,
        )


def test_save_edge_to_npy_raises_without_accepted_detections(tmp_path, video):
    """Test that saving fails when no detection passed quality control."""
    video.detections.append(EdgeDetectionFailure("failure"))

    with pytest.raises(ValueError, match="no accepted edge detections"):
        video.save_edge_to_npy(tmp_path / "edges")
