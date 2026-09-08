"""Regression tests for the shared radial baseline implementation."""
import numpy as np
import pytest

from vesmod.VesEdge.contour_geometry import fit_radial_baseline
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
