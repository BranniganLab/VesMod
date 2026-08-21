"""Unit tests for edge_filtering.py."""

import numpy as np
import pytest

from vesmod.VesEdge.config import EdgeQCConfig
from vesmod.VesEdge.edge_filtering import (
    check_curvature,
    check_edge_populations,
)
from vesmod.VesEdge.models import (
    EdgeDetection,
    EdgeDetectionFailure,
    ImageContour,
    QCFlag,
)


def _make_edge(
    origin=(0.0, 0.0),
    radius=10.0,
    n_samples=8,
):
    """Return a simple successful edge detection for QC tests."""
    radii = np.full(
        n_samples,
        radius,
        dtype=float,
    )

    return EdgeDetection(
        ImageContour(
            origin,
            radii.copy(),
        ),
        ImageContour(
            origin,
            radii.copy(),
        ),
        radii.copy(),
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"curvature_threshold": -1.0},
            "curvature_threshold must be non-negative",
        ),
        (
            {"population_bic_threshold": np.nan},
            "population_bic_threshold must be finite",
        ),
        (
            {"max_minor_population_fraction": 0.5},
            "max_minor_population_fraction must be greater than or equal to 0 and less than 0.5",
        ),
    ],
)
def test_edge_qc_config_rejects_invalid_values(
    kwargs,
    match,
):
    """Test representative invalid QC configuration values."""
    config_values = {
        "curvature_threshold": 5.0,
        "population_bic_threshold": 10.0,
        "max_minor_population_fraction": 0.25,
    }
    config_values.update(kwargs)

    with pytest.raises(
        ValueError,
        match=match,
    ):
        EdgeQCConfig(**config_values)


def test_check_curvature_accepts_smooth_contour():
    """Test that a constant-radius contour passes curvature QC."""
    edge = _make_edge()

    check_curvature(
        edge,
        threshold=1.0,
    )

    assert edge.qc.curvature_score == pytest.approx(0.0)
    assert QCFlag.CURVATURE not in edge.qc.flags


def test_check_curvature_rejects_large_local_deviation():
    """Test that a sharp radial deviation fails curvature QC."""
    edge = _make_edge()
    edge.analysis_contour.r[3] = 20.0

    check_curvature(
        edge,
        threshold=1.0,
    )

    assert edge.qc.curvature_score >= 1.0
    assert QCFlag.CURVATURE in edge.qc.flags


def test_check_curvature_accepts_score_equal_to_threshold():
    """Test that a curvature score equal to the threshold passes QC."""
    edge = _make_edge()
    edge.analysis_contour.r[3] = 20.0

    check_curvature(
        edge,
        threshold=np.finfo(float).max,
    )
    threshold = edge.qc.curvature_score

    check_curvature(
        edge,
        threshold=threshold,
    )

    assert QCFlag.CURVATURE not in edge.qc.flags


@pytest.mark.parametrize(
    (
        "bic_threshold",
        "max_minor_fraction",
        "match",
    ),
    [
        (
            np.nan,
            0.25,
            "bic_threshold must be finite",
        ),
        (
            -1.0,
            0.25,
            "bic_threshold must be non-negative",
        ),
        (
            10.0,
            0.5,
            "max_minor_fraction",
        ),
    ],
)
def test_check_edge_populations_rejects_invalid_parameters(
    bic_threshold,
    max_minor_fraction,
    match,
):
    """Test validation of parameters accepted directly by population QC."""
    with pytest.raises(
        ValueError,
        match=match,
    ):
        check_edge_populations(
            [],
            bic_threshold=bic_threshold,
            max_minor_fraction=max_minor_fraction,
        )


def test_check_edge_populations_short_circuits_for_too_few_edges():
    """Test that fewer than eight usable edges are treated as one population."""
    edges = [
        _make_edge(origin=(float(index), 0.0))
        for index in range(7)
    ]

    result = check_edge_populations(
        edges,
        bic_threshold=10.0,
        max_minor_fraction=0.25,
    )

    assert result.bic_one_population is None
    assert result.bic_two_populations is None
    assert not result.two_populations_detected
    assert result.population_sizes == (7,)
    assert result.rejected_population is None

    assert all(
        edge.qc.population_label == 0
        for edge in edges
    )
    assert all(
        edge.qc.population_probability == 1.0
        for edge in edges
    )


def test_check_edge_populations_uses_only_edges_that_pass_preceding_qc():
    """Test that failures and previous QC rejections are excluded."""
    usable_edges = [
        _make_edge(
            origin=(float(index), 0.0)
        )
        for index in range(3)
    ]

    rejected_edge = _make_edge(
        origin=(10.0, 10.0)
    )
    rejected_edge.qc.flags.add(
        QCFlag.CURVATURE
    )

    detections = [
        *usable_edges,
        rejected_edge,
        EdgeDetectionFailure("failure"),
    ]

    result = check_edge_populations(
        detections,
        bic_threshold=10.0,
        max_minor_fraction=0.25,
    )

    assert result.population_sizes == (3,)
    assert rejected_edge.qc.population_label is None
    assert rejected_edge.qc.population_probability is None


def test_check_edge_populations_keeps_one_population_when_bic_gain_is_insufficient():
    """Test that one population is retained when BIC improvement is too small."""
    edges = [
        _make_edge(
            origin=(
                0.1 * index,
                -0.1 * index,
            ),
            radius=10.0 + 0.05 * index,
        )
        for index in range(10)
    ]

    result = check_edge_populations(
        edges,
        bic_threshold=1e9,
        max_minor_fraction=0.25,
    )

    assert not result.two_populations_detected
    assert result.population_sizes == (10,)
    assert result.rejected_population is None
    assert result.bic_one_population is not None
    assert result.bic_two_populations is not None

    assert all(
        edge.qc.population_label == 0
        for edge in edges
    )
    assert all(
        edge.qc.population_probability == 1.0
        for edge in edges
    )
    assert all(
        QCFlag.POPULATION_OUTLIER not in edge.qc.flags
        for edge in edges
    )


def test_check_edge_populations_rejects_small_separate_population():
    """Test that a well-separated minor population is rejected."""
    major_edges = [
        _make_edge(
            origin=(
                0.05 * index,
                -0.05 * index,
            ),
            radius=10.0 + 0.02 * index,
        )
        for index in range(8)
    ]

    minor_edges = [
        _make_edge(
            origin=(
                20.0 + 0.05 * index,
                20.0 - 0.05 * index,
            ),
            radius=30.0 + 0.02 * index,
        )
        for index in range(2)
    ]

    result = check_edge_populations(
        major_edges + minor_edges,
        bic_threshold=0.0,
        max_minor_fraction=0.25,
    )

    assert result.two_populations_detected
    assert sorted(result.population_sizes) == [2, 8]
    assert result.rejected_population is not None

    assert all(
        QCFlag.POPULATION_OUTLIER not in edge.qc.flags
        for edge in major_edges
    )
    assert all(
        QCFlag.POPULATION_OUTLIER in edge.qc.flags
        for edge in minor_edges
    )

    assert all(
        edge.qc.population_label is not None
        for edge in major_edges + minor_edges
    )
    assert all(
        0.0 <= edge.qc.population_probability <= 1.0
        for edge in major_edges + minor_edges
    )


def test_check_edge_populations_does_not_reject_large_second_population():
    """Test that a second population above the size cutoff is retained."""
    first_population = [
        _make_edge(
            origin=(
                0.05 * index,
                -0.05 * index,
            ),
            radius=10.0 + 0.02 * index,
        )
        for index in range(6)
    ]

    second_population = [
        _make_edge(
            origin=(
                20.0 + 0.05 * index,
                20.0 - 0.05 * index,
            ),
            radius=30.0 + 0.02 * index,
        )
        for index in range(4)
    ]

    edges = (
        first_population
        + second_population
    )

    result = check_edge_populations(
        edges,
        bic_threshold=0.0,
        max_minor_fraction=0.25,
    )

    assert result.two_populations_detected
    assert sorted(result.population_sizes) == [4, 6]
    assert result.rejected_population is None

    assert all(
        QCFlag.POPULATION_OUTLIER not in edge.qc.flags
        for edge in edges
    )
