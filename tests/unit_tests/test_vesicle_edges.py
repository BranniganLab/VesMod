"""Unit tests for vesicle_edges.py."""

import numpy as np
import pytest

from vesmod.VesEdge import (
    EdgeExtractionConfig,
    EdgeQCConfig,
    QCFlag,
    VesicleEdges,
)
from vesmod.VesEdge.models import (
    EdgeDetection,
    EdgeDetectionFailure,
    ImageContour,
)


def _edge(
    radius=10.0,
    origin=(0.0, 0.0),
    n_samples=8,
):
    """Return a simple successful detection."""
    radii = np.full(n_samples, radius, dtype=float)
    contour = ImageContour(origin, radii.copy())
    return EdgeDetection(contour, contour)


@pytest.fixture
def extraction_config():
    """Return standard extraction settings."""
    return EdgeExtractionConfig(
        pixels_per_micron=2.0,
        n_angular_samples=8,
    )


@pytest.fixture
def qc_config():
    """Return permissive frame-level QC settings."""
    return EdgeQCConfig(
        curvature_threshold=100.0,
    )


@pytest.fixture
def edges(extraction_config):
    """Return extracted edges before QC."""
    return VesicleEdges(
        extraction_config=extraction_config,
        detections=[_edge(), _edge(radius=11.0)],
    )


def test_successful_detections_excludes_failures(extraction_config):
    """Test successful_detections filters extraction failures."""
    first = _edge()
    second = _edge(radius=12.0)
    edge_results = VesicleEdges(
        extraction_config=extraction_config,
        detections=[
            first,
            EdgeDetectionFailure("failure"),
            second,
        ],
    )

    assert edge_results.successful_detections == [first, second]


def test_post_init_backfills_legacy_frame_indices(extraction_config):
    """Test legacy/manual results gain source indices from stored order."""
    edge_results = VesicleEdges(
        extraction_config=extraction_config,
        detections=[
            _edge(),
            EdgeDetectionFailure("failure"),
            _edge(radius=12.0),
        ],
    )

    assert [result.frame_index for result in edge_results.detections] == [0, 1, 2]


def test_post_init_rejects_inconsistent_frame_index(extraction_config):
    """Test explicit source indices must agree with trajectory position."""
    detection = _edge()
    detection.frame_index = 3

    with pytest.raises(ValueError, match="frame_index"):
        VesicleEdges(
            extraction_config=extraction_config,
            detections=[detection],
        )


def test_accepted_detections_requires_qc(edges):
    """Test that fresh extraction results are not implicitly accepted."""
    with pytest.raises(ValueError, match="Quality control has not been run"):
        _ = edges.accepted_detections


def test_run_qc_requires_configuration(edges):
    """Test that QC cannot run without explicit or previously stored settings."""
    with pytest.raises(ValueError, match="configuration is required"):
        edges.run_qc()


def test_run_qc_records_aggregate_results(edges, qc_config):
    """Test that a completed QC run records config and enabled QC results."""
    edges.run_qc(qc_config)

    assert edges.qc_result is not None
    assert edges.qc_result.config is qc_config
    assert edges.qc_config is qc_config
    assert edges.qc_result.curvature is not None
    assert len(edges.qc_result.curvature.scores) == 2
    assert edges.qc_result.curvature.rejected_count == 0


def test_run_qc_preserves_frame_indices(edges, qc_config):
    """Test QC does not alter source-frame identity."""
    original_indices = [result.frame_index for result in edges.detections]

    edges.run_qc(qc_config)

    assert [result.frame_index for result in edges.detections] == original_indices


def test_run_qc_can_recover_previous_curvature_rejection(extraction_config):
    """Test that rerunning QC can recover a previously rejected detection."""
    detection = _edge(n_samples=20)
    detection.analysis_contour.r[5] = 30.0
    edge_results = VesicleEdges(
        extraction_config=EdgeExtractionConfig(
            pixels_per_micron=1.0,
            n_angular_samples=20,
        ),
        detections=[detection],
    )
    strict = EdgeQCConfig(
        curvature_threshold=1.0,
    )
    permissive = EdgeQCConfig(
        curvature_threshold=100.0,
    )

    with pytest.raises(ValueError, match="no frames passed quality control"):
        edge_results.run_qc(strict)
    assert edge_results.qc_config is strict
    assert edge_results.qc_result is not None
    assert edge_results.qc_result.curvature is not None
    assert edge_results.qc_result.curvature.rejected_count == 1
    assert QCFlag.CURVATURE in detection.qc.flags

    edge_results.run_qc(permissive)

    assert edge_results.qc_result is not None
    assert edge_results.qc_result.curvature is not None
    assert edge_results.qc_result.curvature.rejected_count == 0
    assert QCFlag.CURVATURE not in detection.qc.flags
    assert detection.qc.passed


def test_run_qc_clears_stale_qc_state(edges, qc_config):
    """Test that a new QC run discards prior per-detection QC state."""
    first = edges.successful_detections[0]
    first.qc.flags.add(QCFlag.CURVATURE)
    first.qc.curvature_score = 999.0

    edges.run_qc(qc_config)

    assert edges.qc_result is not None
    assert first.qc.curvature_score == pytest.approx(0.0)
    assert QCFlag.CURVATURE not in first.qc.flags


def test_save_edge_to_npy_requires_qc(tmp_path, edges):
    """Test that un-QCed extraction results cannot be exported as accepted."""
    with pytest.raises(ValueError, match="Quality control has not been run"):
        edges.save_edge_to_npy(tmp_path / "edges.npy")


def test_save_edge_to_npy_saves_only_accepted_in_microns(
    tmp_path,
    extraction_config,
):
    """Test filtered NumPy export converts accepted radii to microns."""
    accepted = _edge(radius=10.0)
    rejected = _edge(radius=10.0)
    rejected.analysis_contour.r[3] = 30.0
    edge_results = VesicleEdges(
        extraction_config=extraction_config,
        detections=[accepted, rejected],
    )
    config = EdgeQCConfig(
        curvature_threshold=1.0,
        enable_area_qc=False,
    )
    edge_results.run_qc(config)

    edge_results.save_edge_to_npy(tmp_path / "filtered")
    saved = np.load(tmp_path / "filtered.npy")

    assert saved.shape == (1, 8)
    np.testing.assert_array_equal(
        saved[0],
        accepted.analysis_contour.r / 2.0,
    )


def test_checkpoint_round_trip_preserves_extraction_not_qc(
    tmp_path,
    extraction_config,
    qc_config,
):
    """Test checkpointing preserves extraction state and source provenance."""
    first = _edge(radius=10.0)
    second = _edge(radius=12.0)
    source_path = tmp_path / "source.nd2"
    edge_results = VesicleEdges(
        extraction_config=extraction_config,
        detections=[
            first,
            EdgeDetectionFailure("bad frame"),
            second,
        ],
        source_path=source_path,
    )
    edge_results.run_qc(qc_config)
    edge_results.save_checkpoint(tmp_path / "sample")

    loaded = VesicleEdges.from_checkpoint(tmp_path / "sample.npz")

    assert loaded.qc_result is None
    assert loaded.qc_config is None
    assert loaded.extraction_config == extraction_config
    assert loaded.source_path == source_path
    assert isinstance(loaded.detections[0], EdgeDetection)
    assert isinstance(loaded.detections[1], EdgeDetectionFailure)
    assert loaded.detections[1].error == "bad frame"
    assert isinstance(loaded.detections[2], EdgeDetection)
    assert [result.frame_index for result in loaded.detections] == [0, 1, 2]
    assert all(
        detection.qc.curvature_score is None
        for detection in loaded.successful_detections
    )
    np.testing.assert_array_equal(
        loaded.successful_detections[0].analysis_contour.r,
        first.analysis_contour.r,
    )


def test_checkpoint_without_optional_provenance_infers_available_data(
    tmp_path,
    extraction_config,
):
    """Test older checkpoints infer frame indices and omit unknown source path."""
    edge_results = VesicleEdges(
        extraction_config=extraction_config,
        detections=[
            _edge(radius=10.0),
            EdgeDetectionFailure("bad frame"),
            _edge(radius=12.0),
        ],
        source_path=tmp_path / "source.nd2",
    )
    current_path = tmp_path / "current.npz"
    edge_results.save_checkpoint(current_path)

    with np.load(current_path, allow_pickle=False) as checkpoint:
        legacy_data = {
            key: checkpoint[key].copy()
            for key in checkpoint.files
            if key not in {"frame_indices", "source_path"}
        }
    legacy_path = tmp_path / "legacy.npz"
    np.savez(legacy_path, **legacy_data)

    loaded = VesicleEdges.from_checkpoint(legacy_path)

    assert [result.frame_index for result in loaded.detections] == [0, 1, 2]
    assert loaded.source_path is None
    assert isinstance(loaded.detections[1], EdgeDetectionFailure)
    assert loaded.detections[1].error == "bad frame"


def test_checkpoint_preserves_variable_native_lengths(tmp_path):
    """Test native contours may differ when analysis lengths match."""
    config = EdgeExtractionConfig(
        pixels_per_micron=2.0,
        n_angular_samples=4,
    )
    first = EdgeDetection(
        ImageContour((1.0, 2.0), np.arange(6, dtype=float) + 1),
        ImageContour((1.0, 2.0), np.arange(4, dtype=float) + 1),
    )
    second = EdgeDetection(
        ImageContour((3.0, 4.0), np.arange(9, dtype=float) + 2),
        ImageContour((3.0, 4.0), np.arange(4, dtype=float) + 2),
    )
    edge_results = VesicleEdges(config, [first, second])

    edge_results.save_checkpoint(tmp_path / "variable.npz")
    loaded = VesicleEdges.from_checkpoint(tmp_path / "variable.npz")

    assert loaded.successful_detections[0].full_contour.r.shape == (6,)
    assert loaded.successful_detections[1].full_contour.r.shape == (9,)
    np.testing.assert_array_equal(
        loaded.successful_detections[0].full_contour.r,
        first.full_contour.r,
    )
    np.testing.assert_array_equal(
        loaded.successful_detections[1].full_contour.r,
        second.full_contour.r,
    )


def test_checkpoint_rejects_differing_contour_origins(tmp_path):
    """Test checkpoints do not silently discard a distinct analysis origin."""
    config = EdgeExtractionConfig(
        pixels_per_micron=2.0,
        n_angular_samples=4,
    )
    detection = EdgeDetection(
        ImageContour((1.0, 2.0), np.ones(6)),
        ImageContour((3.0, 4.0), np.ones(4)),
    )
    edge_results = VesicleEdges(config, [detection])

    with pytest.raises(ValueError, match="full and analysis contour origins differ"):
        edge_results.save_checkpoint(tmp_path / "invalid.npz")


def test_from_checkpoint_can_be_re_qced_with_new_settings(
    tmp_path,
    extraction_config,
):
    """Test loaded extraction results can be evaluated under a new QC config."""
    detection = _edge()
    edge_results = VesicleEdges(extraction_config, [detection])
    edge_results.save_checkpoint(tmp_path / "sample.npz")

    loaded = VesicleEdges.from_checkpoint(tmp_path / "sample.npz")
    new_config = EdgeQCConfig(
        curvature_threshold=100.0,
    )
    loaded.run_qc(new_config)

    assert loaded.qc_result is not None
    assert loaded.qc_result.config is new_config
    assert len(loaded.accepted_detections) == 1


def test_run_qc_applies_area_deviation_and_records_summary(extraction_config):
    """Test trajectory QC rejects a substantially smaller contour."""
    edge_results = VesicleEdges(
        extraction_config=extraction_config,
        detections=[
            _edge(radius=10.0),
            _edge(radius=10.0),
            _edge(radius=5.0),
        ],
    )

    edge_results.run_qc(
        EdgeQCConfig(
            curvature_threshold=100.0,
            max_relative_area_deviation=0.25,
        )
    )

    assert edge_results.qc_result.area is not None
    assert edge_results.qc_result.area.rejected_count == 1
    assert edge_results.qc_result.area.reference_area_pixels2 == pytest.approx(
        100.0 * np.pi
    )
    assert QCFlag.AREA_DEVIATION in edge_results.detections[2].qc.flags
    assert edge_results.accepted_detections == edge_results.detections[:2]


def test_run_qc_can_disable_area_deviation(extraction_config):
    """Test disabling area QC preserves otherwise acceptable contours."""
    edge_results = VesicleEdges(
        extraction_config=extraction_config,
        detections=[
            _edge(radius=10.0),
            _edge(radius=10.0),
            _edge(radius=2.0),
        ],
    )

    edge_results.run_qc(
        EdgeQCConfig(
            curvature_threshold=100.0,
            enable_area_qc=False,
        )
    )

    assert edge_results.qc_result.area is None
    assert len(edge_results.accepted_detections) == 3
    assert all(
        detection.qc.area_pixels2 is None
        for detection in edge_results.successful_detections
    )
