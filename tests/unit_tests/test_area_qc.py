"""Tests for trajectory-level contour-area deviation QC."""

import numpy as np
import pytest

from vesmod.VesEdge.models import EdgeDetection, ImageContour
from vesmod.VesEdge.contour_geometry import contour_area
from vesmod.VesEdge.qc.experimental.area_qc import check_area_deviation


def _edge_with_area(area_pixels2, frame_index=0):
    """Return a circular edge enclosing the requested area."""
    radius = np.sqrt(area_pixels2 / np.pi)
    contour = ImageContour((0.0, 0.0), np.full(16, radius))
    return EdgeDetection(contour, contour, frame_index=frame_index)


def test_contour_area_uses_mean_squared_radius():
    """Test noncircular radial variation is retained in the area integral."""
    radii = np.asarray([1.0, 3.0, 1.0, 3.0])

    area = contour_area(radii)

    assert area == pytest.approx(5.0 * np.pi)
    assert area != pytest.approx(np.pi * np.mean(radii) ** 2)


def test_area_qc_uses_trajectory_median_and_flags_both_directions():
    """Test smaller and larger contour-area deviations are symmetric."""
    detections = [
        _edge_with_area(100.0),
        _edge_with_area(102.0),
        _edge_with_area(40.0),
        _edge_with_area(180.0),
        _edge_with_area(98.0),
    ]

    for index, edge in enumerate(detections):
        edge.frame_index = index
    result = check_area_deviation(
        detections,
        max_relative_deviation=0.25,
    )

    assert result.extras["reference_area_pixels2"] == pytest.approx(100.0)
    assert result.rejected_count == 2
    assert result.rejected[2]
    assert result.rejected[3]
    assert not result.rejected[0]


def test_area_qc_accepts_detection_exactly_on_boundary():
    """Test deviation equal to the configured maximum remains accepted."""
    detections = [
        _edge_with_area(80.0),
        _edge_with_area(100.0),
        _edge_with_area(120.0),
    ]

    for index, edge in enumerate(detections):
        edge.frame_index = index
    result = check_area_deviation(
        detections,
        max_relative_deviation=0.20,
    )

    assert result.rejected_count == 0
    assert not any(result.rejected)



def test_area_qc_evaluates_every_frame_without_another_checks_decisions():
    detections = [_edge_with_area(area, index) for index, area in enumerate((400, 100, 102))]
    result = check_area_deviation(detections, max_relative_deviation=0.25)
    assert result.extras["reference_area_pixels2"] == pytest.approx(102.0)
    assert result.rejected == (True, False, False)
    assert all(not hasattr(edge, "qc") for edge in detections)


def test_area_qc_rejects_frames_without_a_finite_positive_reference():
    detections = [_edge_with_area(100, index) for index in range(2)]
    for edge in detections:
        object.__setattr__(edge.full_contour, "r", np.full(16, np.nan))
    result = check_area_deviation(detections, max_relative_deviation=0.25)
    assert result.extras["reference_area_pixels2"] is None
    assert result.diagnostics["relative_area_deviation"] == (None, None)
    assert result.rejected == (True, True)
