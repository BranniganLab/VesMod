"""Unit tests for VesicleVideo checkpoint serialization."""

import numpy as np
import pytest

from vesmod.VesEdge.edge_filtering import EdgeQCConfig
from vesmod.VesEdge.models import (
    EdgeDetection,
    EdgeDetectionFailure,
)
from vesmod.VesEdge.vesicle_video import (
    EdgeExtractionConfig,
    VesicleVideo,
)


def _qc_config(
    curvature_threshold=100.0,
):
    """Return QC settings that disable trajectory population analysis."""
    return EdgeQCConfig(
        curvature_threshold=curvature_threshold,
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        enable_population_qc=False,
    )


def test_checkpoint_round_trip_preserves_extraction_results(tmp_path):
    """Test that checkpoints preserve detections, failures, and contour data."""
    extraction_config = EdgeExtractionConfig(
        pixels_per_micron=2.0,
        n_angular_samples=4,
    )
    video = VesicleVideo(
        np.zeros((3, 20, 20)),
        extraction_config,
        _qc_config(),
    )

    first = video._compile_edge_detection_results(
        np.arange(6, dtype=float) + 10.0,
        (8.0, 7.0),
    )
    second = video._compile_edge_detection_results(
        np.arange(8, dtype=float) + 20.0,
        (9.0, 6.0),
    )
    video.detections = [
        first,
        EdgeDetectionFailure("intentional failure"),
        second,
    ]

    checkpoint_path = tmp_path / "video_state"
    video.save_checkpoint(checkpoint_path)

    loaded = VesicleVideo.from_checkpoint(
        checkpoint_path.with_suffix(".npz")
    )

    assert loaded.frames is None
    assert loaded.extraction_config == extraction_config
    assert len(loaded.detections) == 3
    assert isinstance(loaded.detections[0], EdgeDetection)
    assert isinstance(loaded.detections[1], EdgeDetectionFailure)
    assert isinstance(loaded.detections[2], EdgeDetection)
    assert loaded.detections[1].error == "intentional failure"

    loaded_first = loaded.detections[0]
    loaded_second = loaded.detections[2]
    assert loaded_first.full_contour.origin == first.full_contour.origin
    assert loaded_second.full_contour.origin == second.full_contour.origin
    np.testing.assert_array_equal(
        loaded_first.full_contour.r,
        first.full_contour.r,
    )
    np.testing.assert_array_equal(
        loaded_second.full_contour.r,
        second.full_contour.r,
    )
    np.testing.assert_array_equal(
        loaded_first.analysis_contour.r,
        first.analysis_contour.r,
    )
    np.testing.assert_array_equal(
        loaded_second.analysis_contour.r,
        second.analysis_contour.r,
    )
    np.testing.assert_array_equal(
        loaded_first.radii_microns,
        first.radii_microns,
    )
    np.testing.assert_array_equal(
        loaded_second.radii_microns,
        second.radii_microns,
    )


def test_from_checkpoint_can_apply_new_qc_settings(tmp_path):
    """Test that a checkpoint can recover an edge rejected by earlier QC."""
    extraction_config = EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=None,
    )
    strict_qc = _qc_config(
        curvature_threshold=1.0,
    )
    video = VesicleVideo(
        np.zeros((1, 20, 20)),
        extraction_config,
        strict_qc,
    )
    radii = np.full(8, 10.0)
    radii[3] = 20.0
    video.detections = [
        video._compile_edge_detection_results(
            radii,
            (10.0, 10.0),
        )
    ]

    with pytest.raises(
        ValueError,
        match="no frames passed quality control",
    ):
        video.run_qc()

    checkpoint_path = tmp_path / "strict_state.npz"
    video.save_checkpoint(checkpoint_path)

    loaded = VesicleVideo.from_checkpoint(
        checkpoint_path,
        qc_config=_qc_config(
            curvature_threshold=100.0,
        ),
    )

    assert loaded.qc_config.curvature_threshold == 100.0
    assert isinstance(loaded.detections[0], EdgeDetection)
    assert loaded.detections[0].accepted


def test_checkpoint_loaded_video_rejects_image_only_operations(tmp_path):
    """Test that checkpoint-only videos cannot extract edges or render GIFs."""
    video = VesicleVideo(
        np.zeros((1, 20, 20)),
        EdgeExtractionConfig(
            pixels_per_micron=1.0,
            n_angular_samples=None,
        ),
        _qc_config(),
    )
    video.detections = [
        video._compile_edge_detection_results(
            np.full(8, 10.0),
            (10.0, 10.0),
        )
    ]
    checkpoint_path = tmp_path / "state.npz"
    video.save_checkpoint(checkpoint_path)
    loaded = VesicleVideo.from_checkpoint(checkpoint_path)

    with pytest.raises(
        ValueError,
        match="image frames are not available",
    ):
        loaded.extract_edges(
            lambda frame: (np.full(8, 10.0), (10.0, 10.0))
        )

    with pytest.raises(
        ValueError,
        match="image frames are not available",
    ):
        loaded.make_vesicle_gif(
            tmp_path / "loaded.gif"
        )


def test_checkpoint_loaded_video_can_save_new_npy_output(tmp_path):
    """Test that reloaded checkpoint data can produce a renamed npy output."""
    video = VesicleVideo(
        np.zeros((1, 20, 20)),
        EdgeExtractionConfig(
            pixels_per_micron=2.0,
            n_angular_samples=None,
        ),
        _qc_config(),
    )
    video.detections = [
        video._compile_edge_detection_results(
            np.full(8, 20.0),
            (10.0, 10.0),
        )
    ]
    checkpoint_path = tmp_path / "state.npz"
    video.save_checkpoint(checkpoint_path)

    loaded = VesicleVideo.from_checkpoint(checkpoint_path)
    output_path = tmp_path / "requc_output.npy"
    loaded.save_edge_to_npy(output_path)

    saved = np.load(output_path)
    assert saved.shape == (1, 8)
    np.testing.assert_allclose(saved[0], 10.0)
