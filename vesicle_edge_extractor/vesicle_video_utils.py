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
    theta = np.linspace(0, 2 * np.pi, num_r)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_vals = r_vals * cos_theta + origin_x
    y_vals = r_vals * sin_theta + origin_y
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


def isolate_region_of_array(arr, mask_center, threshold, set_bg_to_nan=False):
    """
    Use a mask to preserve region of array within threshold of mask_center.

    If mask_center is a scalar, preserve within threshold of mask_center on all
    rows. If mask_center is a list or ndarray, preserve within threshold of
    mask_center[i] on each row i.

    Parameters
    ----------
    arr : 2D numpy ndarray
        The original image/array that you wish to mask.
    mask_center : int, float, list or numpy ndarray
        The center column index from which to mask. If a scalar is provided,
        use a static mask over all rows. If a list or ndarray is provided, use
        a moving mask over each row.
    threshold : int
        How many bins on either side of mask_center to preserve.
    set_bg_to_nan : bool, OPTIONAL
        If True, set all values outside of mask to np.nan. If False, set all
        values outside of mask to 0.

    Raises
    ------
    IndexError
        If arr is not a 2D array or if arr and mask_center are different sizes
        along 0th dimension.
    TypeError
        If arr isn't a numpy array or if mask_center isn't an scalar, array or
        list.

    Returns
    -------
    masked_copy : 2D numpy ndarray
        The original arr with all values set to 0 or np.nan except those
        within the masked region.

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
        lower_bound = int(mask_center - mask_center * threshold)
        upper_bound = int(mask_center + mask_center * threshold) + 1
        masked_copy[:, lower_bound:upper_bound] = arr[:, lower_bound:upper_bound]
    elif isinstance(mask_center, (np.ndarray, list)):
        # moving mask: preserve within threshold of mask_center[i] on row i.
        if isinstance(mask_center, list):
            mask_center = np.array(mask_center)
        if mask_center.shape[0] != arr.shape[0]:
            raise IndexError("arr and mask_center must be same size in 0th dimension")
        for index, center_value in enumerate(mask_center):
            lower_bound = int(center_value - center_value * threshold)
            upper_bound = int(center_value + center_value * threshold) + 1
            masked_copy[index, lower_bound:upper_bound] = arr[index, lower_bound:upper_bound]
    else:
        raise TypeError("mask_center must be a scalar, list, or numpy ndarray.")

    return masked_copy


def measure_second_derivative(arr):
    """
    Wrap the array and measure its second derivative.

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
