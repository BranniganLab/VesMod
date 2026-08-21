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
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        enable_population_qc=False,
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


def test_accepted_detections_requires_qc(edges):
    """Test that fresh extraction results are not implicitly accepted."""
    with pytest.raises(ValueError, match="Quality control has not been run"):
        _ = edges.accepted_detections


def test_run_qc_requires_configuration(edges):
    """Test that QC cannot run without explicit or previously stored settings."""
    with pytest.raises(ValueError, match="configuration is required"):
        edges.run_qc()


def test_run_qc_updates_configuration(edges, qc_config):
    """Test that QC records the configuration used for completed results."""
    edges.run_qc(qc_config)
    assert edges.qc_config is qc_config


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
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        enable_population_qc=False,
    )
    permissive = EdgeQCConfig(
        curvature_threshold=100.0,
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        enable_population_qc=False,
    )

    with pytest.raises(ValueError, match="no frames passed quality control"):
        edge_results.run_qc(strict)
    assert edge_results.qc_config is strict
    assert QCFlag.CURVATURE in detection.qc.flags

    edge_results.run_qc(permissive)

    assert QCFlag.CURVATURE not in detection.qc.flags
    assert detection.qc.passed


def test_run_qc_clears_stale_population_state(edges, qc_config):
    """Test that a new QC run discards prior derived trajectory state."""
    first = edges.successful_detections[0]
    first.qc.flags.add(QCFlag.POPULATION_OUTLIER)
    first.qc.population_label = 1
    first.qc.population_probability = 0.9
    edges.population_result = object()

    edges.run_qc(qc_config)

    assert edges.population_result is None
    assert first.qc.population_label is None
    assert first.qc.population_probability is None
    assert QCFlag.POPULATION_OUTLIER not in first.qc.flags


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
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        enable_population_qc=False,
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
    """Test checkpointing preserves raw extraction state, not QC state."""
    first = _edge(radius=10.0)
    second = _edge(radius=12.0)
    edge_results = VesicleEdges(
        extraction_config=extraction_config,
        detections=[
            first,
            EdgeDetectionFailure("bad frame"),
            second,
        ],
    )
    edge_results.run_qc(qc_config)
    edge_results.save_checkpoint(tmp_path / "sample")

    loaded = VesicleEdges.from_checkpoint(tmp_path / "sample.npz")

    assert loaded.qc_config is None
    assert loaded.population_result is None
    assert loaded.extraction_config == extraction_config
    assert isinstance(loaded.detections[0], EdgeDetection)
    assert isinstance(loaded.detections[1], EdgeDetectionFailure)
    assert loaded.detections[1].error == "bad frame"
    assert isinstance(loaded.detections[2], EdgeDetection)
    assert all(
        detection.qc.curvature_score is None
        for detection in loaded.successful_detections
    )
    np.testing.assert_array_equal(
        loaded.successful_detections[0].analysis_contour.r,
        first.analysis_contour.r,
    )


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
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        enable_population_qc=False,
    )
    loaded.run_qc(new_config)

    assert loaded.qc_config is new_config
    assert len(loaded.accepted_detections) == 1
