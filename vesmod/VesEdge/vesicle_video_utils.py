#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 14:27:50 2025.

@author: js2746
"""
import numpy as np
import cv2


def convert_to_cartesian(center_point, r_vals):
    """
    Convert r values to X and Y values.

    Parameters
    ----------
    center_point : tuple
        The origin (in X and Y) of your coordinate system.
    r_vals : list
        The r values to convert

    Returns
    -------
    list
        The X and Y values in cartesian space.

    """
    if not isinstance(r_vals, (list, np.ndarray)):
        raise TypeError("r_vals must be a list or 1D numpy array.")
    if isinstance(r_vals, np.ndarray):
        if len(r_vals.shape) != 1:
            raise TypeError("r_vals cannot have more dimensions than 1.")
        num_r = r_vals.shape[0]
    else:
        num_r = len(r_vals)
    origin_x, origin_y = center_point
    theta = np.linspace(0, 2 * np.pi, num_r, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_vals = r_vals * cos_theta + origin_x
    y_vals = r_vals * sin_theta + origin_y

    # close curve for plotting purposes
    x_vals = np.append(x_vals, x_vals[0])
    y_vals = np.append(y_vals, y_vals[0])
    return x_vals, y_vals


def convert_to_polar(x_vals, y_vals, origin):
    """
    Convert Cartesian coordinates to radius values in polar coordinates.

    Parameters
    ----------
    x_vals : numpy ndarray
        The X coordinates.
    y_vals : numpy ndarray
        The Y coordinates.
    origin : tuple
        The X, Y coordinates of where the origin should be in the polar system.

    Returns
    -------
    numpy ndarray
        1D array of r values. Assumes evenly distributed points in theta from
        0 to 2pi.

    """
    x_vals = x_vals - origin[0]
    y_vals = y_vals - origin[1]
    return np.sqrt(x_vals ** 2 + y_vals ** 2)


def wrap_image_to_polar(image, origin_coords):
    """
    Convert an image from cartesian to polar about an origin point.

    Parameters
    ----------
    image : 2D numpy ndarray
        The image you wish to convert to polar coordinates.
    origin_coords : 2-tuple
        The X and Y coordinates that represent the new image origin.

    Returns
    -------
    2D numpy ndarray
        The image, wrapped into polar coordinates.
    float
        The scaling factor needed to convert back to Cartesian coordinates.

    """
    max_r = np.sqrt(((image.shape[0] / 2.0) ** 2.0) + ((image.shape[1] / 2.0) ** 2.0))
    polar_image = cv2.linearPolar(image, (origin_coords[1], origin_coords[0]), max_r, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)
    scaling_factor = polar_image.shape[1] / max_r
    return polar_image, scaling_factor


def zero_out_all_but_lowest_n_modes(arr, n):
    """
    Take the FFT of a 1d array, remove the lowest n modes, then IFFT.

    Parameters
    ----------
    arr : 1D numpy ndarray or list
        The input array.
    n : int
        The highest mode you wish to retain.

    Raises
    ------
    TypeError
        n must be an int.
    ValueError
        n must be positive.
    IndexError
        n can't exceed the number of positive modes.

    Returns
    -------
    ifft : 1D numpy ndarray
        The original input array, but with high frequencies excluded.

    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be a positive integer")
    if n >= arr.shape[0] // 2:
        raise IndexError(f"arr does not have enough modes ({arr.shape[0]}) to zero out all but the lowest {n}.")
    fft = np.fft.fft(arr)
    fft[n + 1:-1 * n] = 0
    ifft = np.fft.ifft(fft)
    return ifft.real


def isolate_region_of_array(arr, mask_center, window_fraction, set_bg_to_nan=False):
    """
    Mask a 2D array to retain only a local region around one or more center positions.

    A copy of ``arr`` is returned in which values outside a window centered on
    ``mask_center`` are replaced with either 0 or ``np.nan``. The half-width of
    the retained window is defined as a fraction of the center position itself:

    ``half_width = center * window_fraction``

    If ``mask_center`` is a scalar, the same mask is applied to every row. If
    ``mask_center`` is a list or ndarray, a separate mask is applied to each
    row using the corresponding center value.

    Parameters
    ----------
    arr : np.ndarray
        Two-dimensional array to be masked.
    mask_center : int, float, list, or np.ndarray
        Center column index (or indices) defining the retained region. A scalar
        applies a static mask to all rows, while a list or ndarray applies a
        row-specific mask.
    window_fraction : float
        Fractional half-width of the retained region. For a center position
        ``c``, values are preserved from

        ``c - c * window_fraction``

        to

        ``c + c * window_fraction``.

        For example, ``window_fraction=0.05`` retains approximately ±5% of the
        center position on either side of the center.
    set_bg_to_nan : bool, optional
        If ``True``, values outside the retained region are set to ``np.nan``.
        Otherwise they are set to 0.

    Returns
    -------
    np.ndarray
        Copy of ``arr`` with values outside the retained region removed.

    Raises
    ------
    TypeError
        If ``arr`` is not a NumPy array or if ``mask_center`` is not a scalar,
        list, or NumPy array.
    IndexError
        If ``arr`` is not two-dimensional or if a row-wise ``mask_center``
        array does not match the number of rows in ``arr``.

    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a numpy ndarray.")
    if len(arr.shape) != 2:
        raise IndexError("arr must be a 2D array")

    bg = 0
    if set_bg_to_nan:
        bg = np.nan
    masked_copy = np.full_like(arr, bg)

    if np.isscalar(mask_center):
        # static mask: preserve within threshold of mask_center for all rows.
        lower_bound = int(mask_center - mask_center * window_fraction)
        upper_bound = int(mask_center + mask_center * window_fraction) + 1
        masked_copy[:, lower_bound:upper_bound] = arr[:, lower_bound:upper_bound]
    elif isinstance(mask_center, (np.ndarray, list)):
        # moving mask: preserve within threshold of mask_center[i] on row i.
        if isinstance(mask_center, list):
            mask_center = np.array(mask_center)
        if mask_center.shape[0] != arr.shape[0]:
            raise IndexError("arr and mask_center must be same size in 0th dimension")
        for index, center_value in enumerate(mask_center):
            lower_bound = int(center_value - center_value * window_fraction)
            upper_bound = int(center_value + center_value * window_fraction) + 1
            masked_copy[index, lower_bound:upper_bound] = arr[index, lower_bound:upper_bound]
    else:
        raise TypeError("mask_center must be a scalar, list, or numpy ndarray.")

    return masked_copy


def measure_wrapped_finite_second_difference(arr):
    """
    Wrap the array and measure its second difference.

    Parameters
    ----------
    arr : numpy ndarray
        1D array that you wish to take the second derivative of.

    Returns
    -------
    second_deriv : numpy ndarray
        The second derivative (curvature) of the input array.

    """
    wrapped_array = np.pad(arr, pad_width=2, mode='wrap')
    second_deriv = np.diff(wrapped_array, n=2)[1: -1]
    return second_deriv


def downsample_to_new_indices(data: np.ndarray, index_floats: np.ndarray) -> np.ndarray:
    """
    Return values from a 1D array at float-valued indices using linear interpolation.

    The first element is appended to the end so that interpolation can wrap across
    the periodic boundary.
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")
    if index_floats.ndim != 1:
        raise ValueError("Index array must be 1D.")

    if np.any(index_floats < 0) or np.any(index_floats > data.size):
        raise IndexError("One or more indices are out of bounds.")

    wrapped_data = np.append(data, data[0])

    lower_indices = np.floor(index_floats).astype(int)
    upper_indices = np.ceil(index_floats).astype(int)
    weights = index_floats - lower_indices

    return (1 - weights) * wrapped_data[lower_indices] + weights * wrapped_data[upper_indices]
