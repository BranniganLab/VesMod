"""Unit tests for VesEdge data models."""

import numpy as np

from vesmod.VesEdge.models import (
    EdgeDetection,
    EdgeQC,
    ImageContour,
    QCFlag,
)


def test_image_contour_theta_is_evenly_spaced():
    """Test that theta spans one full circle without repeating 2π."""
    contour = ImageContour(
        origin=(0.0, 0.0),
        r=np.ones(4),
    )

    np.testing.assert_allclose(
        contour.theta,
        [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
    )


def test_image_contour_cartesian_coordinates_close_contour():
    """Test conversion from radial to Cartesian coordinates."""
    contour = ImageContour(
        origin=(2.0, 3.0),
        r=np.ones(4),
    )

    np.testing.assert_allclose(
        contour.x,
        [3.0, 2.0, 1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        contour.y,
        [3.0, 4.0, 3.0, 2.0, 3.0],
    )


def test_edge_qc_passed_reflects_flags():
    """Test that an edge passes QC only when it has no failure flags."""
    qc = EdgeQC()

    assert qc.passed

    qc.flags.add(QCFlag.CURVATURE)

    assert not qc.passed


def test_edge_qc_instances_do_not_share_flags():
    """Test that each EdgeQC receives an independent flag set."""
    first = EdgeQC()
    second = EdgeQC()

    first.flags.add(QCFlag.CURVATURE)

    assert QCFlag.CURVATURE in first.flags
    assert QCFlag.CURVATURE not in second.flags


def test_edge_detection_median_radius():
    """Test calculation of the median radius in microns."""
    contour = ImageContour(
        origin=(0.0, 0.0),
        r=np.ones(4),
    )
    detection = EdgeDetection(
        full_contour=contour,
        analysis_contour=contour,
        radii_microns=np.array(
            [1.0, 2.0, 4.0, 8.0],
        ),
    )

    assert detection.median_radius == 3.0


def test_edge_detection_accepted_reflects_qc():
    """Test that accepted delegates to the associated QC result."""
    contour = ImageContour(
        origin=(0.0, 0.0),
        r=np.ones(4),
    )
    detection = EdgeDetection(
        full_contour=contour,
        analysis_contour=contour,
        radii_microns=np.ones(4),
    )

    assert detection.accepted

    detection.qc.flags.add(
        QCFlag.POPULATION_OUTLIER
    )

    assert not detection.accepted