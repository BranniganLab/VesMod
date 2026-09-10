"""Regression tests for shared radial-contour geometry helpers."""
import numpy as np
import pytest

from vesmod.VesEdge.contour_geometry import (
    fit_radial_baseline,
    radial_contour_centroid,
)
from vesmod.VesEdge.vesicle_video_utils import zero_out_all_but_lowest_n_modes


def test_shared_baseline_retains_only_requested_harmonics():
    """The baseline agrees with an independently specified Fourier signal."""
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    expected = 40 + 3 * np.cos(theta) + 1.5 * np.sin(2 * theta)
    radii = expected + 0.4 * np.cos(17 * theta)

    result = fit_radial_baseline(radii, order=7)

    assert result.order == 7
    np.testing.assert_allclose(result.values, expected, rtol=0, atol=1e-12)


def test_shared_baseline_order_zero_returns_mean():
    """A zero-order baseline contains only the mean radius."""
    radii = np.array([1.0, 3.0, 5.0, 7.0])

    result = fit_radial_baseline(radii, order=0)

    np.testing.assert_allclose(result.values, np.full(4, radii.mean()))


@pytest.mark.parametrize("radii", [np.ones((2, 4)), (1.0, 2.0, 3.0)])
def test_shared_baseline_requires_one_dimensional_array_or_list(radii):
    """Inputs without the documented one-dimensional shape are rejected."""
    with pytest.raises(TypeError):
        fit_radial_baseline(radii, order=1)


@pytest.mark.parametrize("order", [1.5, -1, 8])
def test_shared_baseline_rejects_invalid_order(order):
    """Orders must be valid integer harmonics for the input length."""
    with pytest.raises((TypeError, ValueError, IndexError)):
        fit_radial_baseline(np.ones(16), order=order)


def test_radial_contour_centroid_recovers_shifted_circle_center():
    """A circle measured from an offset origin reports its geometric center."""
    n_samples = 720
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    true_center = np.array([3.25, -1.75])
    true_radius = 20.0

    projection = true_center[0] * np.cos(theta) + true_center[1] * np.sin(theta)
    center_distance2 = float(np.dot(true_center, true_center))
    radii = projection + np.sqrt(
        true_radius**2 - center_distance2 + projection**2
    )

    centroid = radial_contour_centroid((0.0, 0.0), radii)

    np.testing.assert_allclose(centroid, true_center, atol=2e-4)


def test_radial_contour_centroid_is_translation_invariant():
    """Translating the radial origin translates the measured centroid equally."""
    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    radii = 18.0 + 1.2 * np.cos(2.0 * theta) + 0.4 * np.sin(5.0 * theta)
    translation = np.array([17.5, -9.25])

    centroid_a = radial_contour_centroid((0.0, 0.0), radii)
    centroid_b = radial_contour_centroid(tuple(translation), radii)

    np.testing.assert_allclose(
        np.asarray(centroid_b) - np.asarray(centroid_a),
        translation,
        rtol=0,
        atol=1e-12,
    )


def test_radial_contour_centroid_rejects_degenerate_contour():
    """A radial representation must contain a polygon with non-zero area."""
    with pytest.raises(ValueError, match="at least three samples"):
        radial_contour_centroid((0.0, 0.0), np.ones(2))


def test_legacy_baseline_wrapper_uses_shared_projection():
    """The existing extraction helper remains numerically compatible."""
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    expected = 40 + 3 * np.cos(theta) + 1.5 * np.sin(2 * theta)
    radii = expected + 0.4 * np.cos(17 * theta)

    np.testing.assert_allclose(
        zero_out_all_but_lowest_n_modes(radii, n=7),
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_legacy_baseline_wrapper_order_zero_preserves_input():
    """The compatibility wrapper retains its legacy order-zero behavior."""
    radii = np.array([1.0, 3.0, 5.0, 7.0])

    result = zero_out_all_but_lowest_n_modes(radii, n=0)

    np.testing.assert_array_equal(result, radii)
