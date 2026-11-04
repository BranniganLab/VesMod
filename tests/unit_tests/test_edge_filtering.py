"""Unit tests for edge_filtering.py."""

import numpy as np
import pytest

from vesmod.VesEdge.qc.checks.edge_filtering import check_curvature
from vesmod.VesEdge.models import EdgeDetection, ImageContour


def _make_edge(
    origin=(0.0, 0.0),
    radius=10.0,
    n_samples=120,
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
        frame_index=0,
    )


def test_check_curvature_accepts_smooth_contour():
    """Test that a constant-radius contour passes curvature QC."""
    edge = _make_edge()
    result = check_curvature([edge], threshold=0.1)

    assert result.diagnostics["curvature_score"][0] == pytest.approx(0.0)
    assert not result.rejected[0]


def test_check_curvature_rejects_large_local_deviation():
    """Test that a sharp radial deviation fails curvature QC."""
    edge = _make_edge()
    edge.analysis_contour.r[3] = 20.0

    result = check_curvature([edge], threshold=0.1)

    assert result.diagnostics["curvature_score"][0] >= 1.0
    assert result.rejected[0]


def test_curvature_score_is_invariant_to_uniform_contour_scaling():
    """Geometrically identical contours have the same normalized score."""
    first = _make_edge(radius=10.0)
    first.analysis_contour.r[3] = 12.0
    second = _make_edge(radius=100.0)
    second.analysis_contour.r[3] = 120.0

    first_result = check_curvature([first], threshold=np.finfo(float).max)
    second_result = check_curvature([second], threshold=np.finfo(float).max)

    assert first_result.diagnostics["curvature_score"][0] == pytest.approx(second_result.diagnostics["curvature_score"][0])


def test_check_curvature_accepts_score_equal_to_threshold():
    """Test that a curvature score equal to the threshold passes QC."""
    edge = _make_edge()
    edge.analysis_contour.r[3] = 20.0

    result = check_curvature(
        [edge],
        threshold=np.finfo(float).max,
    )
    threshold = result.diagnostics["curvature_score"][0]

    result = check_curvature([edge], threshold=threshold)

    assert not result.rejected[0]


def test_curvature_score_is_invariant_to_angular_sampling_resolution():
    """Equivalent contours have matching scores at different resolutions."""
    def make_contour(n_samples):
        theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
        radii = 10.0 * (
            1.0 + 0.02 * np.sin(3.0 * theta)
            + 0.01 * np.cos(7.0 * theta)
        )
        return EdgeDetection(
            ImageContour((0.0, 0.0), radii.copy()),
            ImageContour((0.0, 0.0), radii.copy()),
            frame_index=0,
        )

    scores = []
    for n_samples in (80, 120, 240, 360):
        edge = make_contour(n_samples)
        result = check_curvature([edge], threshold=np.finfo(float).max)
        scores.append(result.diagnostics["curvature_score"][0])

    # The finite-difference approximation converges with resolution, rather
    # than producing bitwise-identical values at every sample count.
    for score in scores[1:]:
        assert score == pytest.approx(scores[0], rel=3e-2)
