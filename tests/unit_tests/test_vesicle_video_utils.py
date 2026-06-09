import numpy as np
import pytest

from vesmod.VesEdge.vesicle_video_utils import (
    convert_to_cartesian,
    convert_to_polar,
    downsample_to_new_indices,
    isolate_region_of_array,
    measure_wrapped_finite_second_difference,
    wrap_image_to_polar,
    zero_out_all_but_lowest_n_modes,
)


def test_convert_to_cartesian_places_points_around_center():
    """Test that radial distances are converted to Cartesian coordinates relative to the supplied center."""
    center = (10.0, 20.0)
    r_vals = np.ones(4)

    x_vals, y_vals = convert_to_cartesian(center, r_vals)

    theta = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    np.testing.assert_allclose(x_vals[:-1], np.cos(theta) + center[0])
    np.testing.assert_allclose(y_vals[:-1], np.sin(theta) + center[1])


def test_convert_to_cartesian_rejects_multidimensional_arrays():
    """Test that convert_to_cartesian rejects numpy arrays with more than one dimension."""
    with pytest.raises(TypeError, match="cannot have more dimensions than 1"):
        convert_to_cartesian((0, 0), np.ones((2, 2)))


def test_convert_to_cartesian_rejects_invalid_types():
    """Test that convert_to_cartesian rejects inputs that are neither lists nor numpy arrays."""
    with pytest.raises(TypeError):
        convert_to_cartesian((0, 0), "bad")


def test_convert_to_polar_recovers_radius():
    """Test that convert_to_polar returns the correct distance from the supplied origin."""
    x_vals = np.array([2, 0, -2])
    y_vals = np.array([0, 2, 0])

    radii = convert_to_polar(x_vals, y_vals, (0, 0))

    np.testing.assert_allclose(radii, np.array([2, 2, 2]))


def test_wrap_image_to_polar_returns_same_shape():
    """Test that wrap_image_to_polar returns an image with the same dimensions as the input image."""
    image = np.zeros((20, 30), dtype=float)

    polar, scale = wrap_image_to_polar(image, (10, 15))

    assert polar.shape == image.shape
    assert scale > 0


def test_zero_out_all_but_lowest_n_modes_preserves_constant_signal():
    """Test that FFT filtering leaves a constant signal unchanged."""
    arr = np.full(16, 7.0)

    filtered = zero_out_all_but_lowest_n_modes(arr, n=2)

    np.testing.assert_allclose(filtered, arr)


def test_zero_out_all_but_lowest_n_modes_accepts_list_input():
    """Test that list input is accepted and converted internally."""
    result = zero_out_all_but_lowest_n_modes([1.0] * 16, n=2)

    np.testing.assert_allclose(result, np.ones(16))


def test_zero_out_all_but_lowest_n_modes_requires_int():
    """Test that n must be an integer."""
    with pytest.raises(TypeError):
        zero_out_all_but_lowest_n_modes(np.ones(16), 2.5)


def test_zero_out_all_but_lowest_n_modes_requires_positive_n():
    """Test that n cannot be negative."""
    with pytest.raises(ValueError):
        zero_out_all_but_lowest_n_modes(np.ones(16), -1)


def test_zero_out_all_but_lowest_n_modes_requires_valid_mode_count():
    """Test that n cannot exceed the available positive Fourier modes."""
    with pytest.raises(IndexError):
        zero_out_all_but_lowest_n_modes(np.ones(16), 8)


def test_isolate_region_of_array_static_mask():
    """Test that a scalar mask center preserves the same columns on every row."""
    arr = np.arange(30).reshape(3, 10)

    masked = isolate_region_of_array(arr, 5, 0.2)

    expected = np.zeros_like(arr)
    expected[:, 4:7] = arr[:, 4:7]

    np.testing.assert_array_equal(masked, expected)


def test_isolate_region_of_array_nan_background():
    """Test that masked background values become NaN when requested."""
    arr = np.ones((3, 10), dtype=float)

    masked = isolate_region_of_array(arr, 5, 0.2, set_bg_to_nan=True)

    assert np.isnan(masked[:, 0]).all()


def test_isolate_region_of_array_moving_mask():
    """Test that each row receives its own mask position when mask_center is an array."""
    arr = np.arange(30).reshape(3, 10)

    masked = isolate_region_of_array(arr, np.array([2, 4, 6]), 0.0)

    assert masked[0, 2] == arr[0, 2]
    assert masked[1, 4] == arr[1, 4]
    assert masked[2, 6] == arr[2, 6]


def test_isolate_region_of_array_requires_matching_lengths():
    """Test that mask_center length must match the number of rows in the input array."""
    with pytest.raises(IndexError):
        isolate_region_of_array(np.ones((3, 10)), np.array([1, 2]), 0.1)


def test_isolate_region_of_array_clips_negative_lower_bound():
    """Test that masks beginning before column zero are clipped to zero.

    This verifies that small mask centers do not trigger negative slicing,
    which would otherwise preserve values from the end of the row instead of
    preserving columns starting at zero.
    """
    arr = np.arange(20).reshape(2, 10)

    masked = isolate_region_of_array(
        arr,
        mask_center=2,
        window_fraction=2.0,
    )

    expected = np.zeros_like(arr)
    expected[:, 0:7] = arr[:, 0:7]

    np.testing.assert_array_equal(masked, expected)


def test_isolate_region_of_array_clips_negative_lower_bound_rowwise():
    """Test that row-specific masks clip negative lower bounds to zero."""
    arr = np.arange(30).reshape(3, 10)
    mask_center = np.array([2, 5, 8])

    masked = isolate_region_of_array(
        arr,
        mask_center=mask_center,
        window_fraction=2.0,
    )

    expected = np.zeros_like(arr)
    expected[0, 0:7] = arr[0, 0:7]
    expected[1, 0:10] = arr[1, 0:10]
    expected[2, 0:10] = arr[2, 0:10]

    np.testing.assert_array_equal(masked, expected)


def test_measure_wrapped_finite_second_difference_returns_zero_for_constant_array():
    """Test that the wrapped second difference of a constant array is zero everywhere."""
    arr = np.ones(10)

    result = measure_wrapped_finite_second_difference(arr)

    np.testing.assert_allclose(result, np.zeros_like(arr))


def test_downsample_to_new_indices_interpolates_correctly():
    """Test that linear interpolation between neighboring samples returns expected values."""
    data = np.array([0, 10, 20, 30])

    result = downsample_to_new_indices(data, np.array([0.5, 1.5]))

    np.testing.assert_allclose(result, np.array([5, 15]))


def test_downsample_to_new_indices_requires_1d_data():
    """Test that input data must be one-dimensional."""
    with pytest.raises(ValueError):
        downsample_to_new_indices(np.ones((2, 2)), np.array([0.5]))


def test_downsample_to_new_indices_requires_1d_indices():
    """Test that interpolation indices must be one-dimensional."""
    with pytest.raises(ValueError):
        downsample_to_new_indices(np.ones(10), np.ones((2, 2)))


def test_downsample_to_new_indices_rejects_out_of_bounds_indices():
    """Test that interpolation indices outside the valid range raise IndexError."""
    with pytest.raises(IndexError):
        downsample_to_new_indices(np.ones(10), np.array([-1]))
