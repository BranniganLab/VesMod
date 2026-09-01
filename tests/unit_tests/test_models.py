"""Unit tests for VesEdge data models."""

import numpy as np
import pytest

from vesmod.VesEdge.models import (
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


def test_image_contour_normalizes_and_copies_inputs():
    """Test successful construction establishes independent float state."""
    radii = np.arange(4, dtype=np.int64) + 1
    contour = ImageContour(
        origin=(np.float64(2.0), np.int64(3)),
        r=radii,
    )

    assert contour.origin == (2.0, 3.0)
    assert all(isinstance(value, float) for value in contour.origin)
    assert contour.r.dtype == float

    radii[0] = 99
    assert contour.r[0] == 1.0


@pytest.mark.parametrize(
    ("origin", "radii", "error_type", "match"),
    [
        ((0.0,), np.ones(4), TypeError, "two-coordinate tuple"),
        ((0.0, np.inf), np.ones(4), ValueError, "origin coordinate must be finite"),
        ((0.0, 0.0), [1.0, 2.0], TypeError, "r must be a NumPy array"),
        ((0.0, 0.0), np.ones((2, 2)), ValueError, "one-dimensional"),
        ((0.0, 0.0), np.array([]), ValueError, "at least one radial sample"),
        ((0.0, 0.0), np.array([1.0, np.nan]), ValueError, "only finite values"),
        ((0.0, 0.0), np.array([1.0, 0.0]), ValueError, "only positive values"),
        (
            (0.0, 0.0),
            np.array([1.0 + 1.0j, 2.0 + 1.0j]),
            TypeError,
            "real-valued",
        ),
    ],
)
def test_image_contour_rejects_invalid_construction_state(
    origin,
    radii,
    error_type,
    match,
):
    """Test malformed contour state is rejected at construction."""
    with pytest.raises(error_type, match=match):
        ImageContour(origin=origin, r=radii)


def test_edge_qc_passed_reflects_flags():
    """Test that QC passes only when it has no failure flags."""
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
