"""Tests for trajectory-level contour-area deviation QC."""

import numpy as np
import pytest

from vesmod.VesEdge import (
    EdgeDetection,
    ImageContour,
    QCFlag,
    check_area_deviation,
    contour_area,
)


def _edge_with_area(area_pixels2):
    """Return a circular edge enclosing the requested area."""
    radius = np.sqrt(area_pixels2 / np.pi)
    contour = ImageContour((0.0, 0.0), np.full(16, radius))
    return EdgeDetection(contour, contour)


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

    result = check_area_deviation(
        detections,
        max_relative_deviation=0.25,
    )

    assert result.reference_area_pixels2 == pytest.approx(100.0)
    assert result.rejected_count == 2
    assert QCFlag.AREA_DEVIATION in detections[2].qc.flags
    assert QCFlag.AREA_DEVIATION in detections[3].qc.flags
    assert QCFlag.AREA_DEVIATION not in detections[0].qc.flags


def test_area_qc_accepts_detection_exactly_on_boundary():
    """Test deviation equal to the configured maximum remains accepted."""
    detections = [
        _edge_with_area(80.0),
        _edge_with_area(100.0),
        _edge_with_area(120.0),
    ]

    result = check_area_deviation(
        detections,
        max_relative_deviation=0.20,
    )

    assert result.rejected_count == 0
    assert all(edge.qc.passed for edge in detections)


def test_area_qc_preserves_existing_qc_flags():
    """Test area QC composes with rather than clears another QC failure."""
    detections = [
        _edge_with_area(100.0),
        _edge_with_area(100.0),
        _edge_with_area(40.0),
    ]
    detections[0].qc.flags.add(QCFlag.CURVATURE)

    check_area_deviation(
        detections,
        max_relative_deviation=0.25,
    )

    assert QCFlag.CURVATURE in detections[0].qc.flags
    assert QCFlag.AREA_DEVIATION in detections[2].qc.flags
