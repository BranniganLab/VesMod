"""Unit tests for edge_filtering.py."""

import numpy as np
import pytest

from vesmod.VesEdge.edge_filtering import check_curvature
from vesmod.VesEdge.models import EdgeDetection, ImageContour, QCFlag


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
        ImageContour(origin, radii.copy()),
        ImageContour(origin, radii.copy()),
    )


def test_check_curvature_accepts_smooth_contour():
    """Test that a constant-radius contour passes curvature QC."""
    edge = _make_edge()
    check_curvature(edge, threshold=1.0)

    assert edge.qc.curvature_score == pytest.approx(0.0)
    assert QCFlag.CURVATURE not in edge.qc.flags


def test_check_curvature_rejects_large_local_deviation():
    """Test that a sharp radial deviation fails curvature QC."""
    edge = _make_edge()
    edge.analysis_contour.r[3] = 20.0

    check_curvature(edge, threshold=1.0)

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

    check_curvature(edge, threshold=threshold)

    assert QCFlag.CURVATURE not in edge.qc.flags
