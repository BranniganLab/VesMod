#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:09:33 2026

@author: js2746
"""

import numpy as np


def read_and_format_csv(path):
    """
    Read in and format the csv.

    Parameters
    ----------
    path : str or Path
        The path to your .csv file.

    Returns
    -------
    ndarray
        A 2D array containing the radii as a function of theta and time.
    float
        The radius of the reference circle.

    """
    csv = np.genfromtxt(path, delimiter=',', skip_header=False, missing_values="NaN", filling_values=np.nan)
    r0 = np.nanmean(csv)
    return csv, r0


def calc_sq_amplitudes(data, norm):
    """
    Take a 2d ndarray and perform FFT along dimension 1. Return squared amplitudes\
    and their associated frequencies.

    Parameters
    ----------
    data : ndarray
        Radius as a function of theta (dimension 1) and time (dimension 0).
    norm : float
        The normalization factor to apply to your amplitudes.

    Returns
    -------
    amps2 : ndarray
        amplitude values for each q, squared.
    freqs : ndarray
        q values.

    """
    # perform fft, square, and average over time
    amps = np.fft.fft(data, axis=1, norm='backward') * norm
    amps2 = amps * amps.conj()

    # get q values
    freqs = np.fft.fftfreq(amps2.shape[1])

    # fftfreq normalizes automatically; un-normalize to get integer q values
    freqs = np.round(freqs * amps2.shape[1]).astype(int)

    return amps2, freqs


def interpolate_indices_vectorized(data: np.ndarray, index_floats: np.ndarray) -> np.ndarray:
    """
    For each row in the 2D `data` array, return the values at each of the float indices\
    provided in `index_floats`, using linear interpolation if necessary.

    Parameters
    ----------
    - data: 2D numpy array of shape (R, C)
    - index_floats: 1D numpy array of shape (N,)

    Returns
    -------
    - 2D numpy array of shape (R, N) with interpolated values
    """
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    if index_floats.ndim != 1:
        raise ValueError("Index array must be 1D.")

    R, C = data.shape
    N = index_floats.shape[0]

    # Check bounds
    if np.any(index_floats < 0) or np.any(index_floats > C):
        raise IndexError("One or more indices are out of bounds.")

    # Wrap first column around to last column
    first_col = data[:, 0]
    first_col = first_col[:, np.newaxis]
    data = np.hstack((data, first_col))

    # Floor and ceil indices
    lower_indices = np.floor(index_floats).astype(int)
    upper_indices = np.ceil(index_floats).astype(int)
    weights = index_floats - lower_indices  # shape: (N,)

    # Broadcast row indices for gathering values
    row_indices = np.arange(R)[:, None]  # shape: (R, 1)

    # Gather values at lower and upper indices
    val_lower = data[row_indices, lower_indices]  # shape: (R, N)
    val_upper = data[row_indices, upper_indices]  # shape: (R, N)

    # Perform linear interpolation
    result = (1 - weights) * val_lower + weights * val_upper  # shape: (R, N)
    assert result.shape == (R, N), f"result not the right shape; expected ({R}, {N}) but got {result.shape}"
    return result


def filter_data(csv_data, filter_type='strict'):
    """
    Move row by row through csv_data. If row passes filter_row(), add it to a \
    new array. At the end, remove any rows of new array that are all zeros or \
    contain a NaN.

    Parameters
    ----------
    csv_data : ndarray
        2D input array to be filtered.
    filter_type : str, optional
        If 'permissive', filter out rows that have NaN values only. If 'strict',
        filter out rows that have NaN values and rows that have curvature aberrations.
        Default is 'strict'.

    Returns
    -------
    filtered_useable_data : ndarray
        Array with filtered rows stripped out.
    filtered_full_data : ndarray
        Array with filtered rows kept in, but set to np.nan.

    """
    assert isinstance(csv_data, np.ndarray), "csv_data must be a numpy array."
    assert len(csv_data.shape) == 2, "csv_data must be a 2D array."
    filtered_useable_data = np.zeros_like(csv_data)
    filtered_full_data = np.zeros_like(csv_data)
    index = 0
    for row in range(csv_data.shape[0]):
        if filter_type == 'permissive':
            if ((filter_row(csv_data[row, :]) == 1) or (filter_row(csv_data[row, :]) == 3)) and (np.nan not in csv_data[row, :]):
                filtered_useable_data[index, :] = csv_data[row, :]
                filtered_full_data[row, :] = csv_data[row, :]
                index += 1
            else:
                filtered_full_data[row, :] = np.nan
        elif filter_type == 'strict':
            if (filter_row(csv_data[row, :]) == 1) and (np.nan not in csv_data[row, :]):
                filtered_useable_data[index, :] = csv_data[row, :]
                filtered_full_data[row, :] = csv_data[row, :]
                index += 1
            else:
                filtered_full_data[row, :] = np.nan
        else:
            raise ValueError("filter_type must be 'strict' or 'permissive'.")

    # remove unused rows at bottom
    filtered_useable_data = filtered_useable_data[:index, :]

    assert csv_data.shape[1] == filtered_useable_data.shape[1], "Something went wrong. The second dimension should not have been resized."
    return filtered_useable_data, filtered_full_data


def filter_row(row_data, test_type='curvature', threshold=5):
    """
    Filter a row of data. If the row is full of NaNs, return 2. If the row \
    contains poorly-segmented values (I.E. there is a big discontinuity from \
    one point to the next), return 3. Otherwise return 1.

    Parameters
    ----------
    row_data : ndarray
        1D ndarray of r values spaced by dtheta.
    test_type : str
        If 'curvature', measure the second derivative wrt theta of all the \
        radii in one row. Filter out any 2nd deriv values above threshold. If \
        'length', measure difference between consecutive radii in row. Filter \
        out any differences higher than threshold*Ri, where Ri is the first \
        radius in the pair. This breaks down with very oblong vesicles and \
        is not sensitive enough to small discontinuities observed in some edge\
        detection cases.
    threshold : float
        If test_type is 'curvature, the absolute curvature value allowable \
        between consecutive radii. If test_type is 'length', the percentage length\
        difference allowable between two consecutive radii. IE the difference \
        between radius b and radius a cannot be greater than (threshold * 100)%\
        of radius a's length.

    Returns
    -------
    int
        1 = healthy, 2 = frame skipped, or 3 = poorly segmented.

    """
    assert isinstance(row_data, np.ndarray), "row_data must be a numpy ndarray"
    assert len(row_data.shape) == 1, "row_data must be a 1d numpy array."
    assert test_type in ['curvature', 'length'], "test_type should be 'curvature' or 'length'."
    assert threshold > 0, "threshold should be a positive number"
    if np.isnan(row_data).any():
        # edge extraction suffered an error and skipped this frame, return 2
        return 2
    else:
        if test_type == 'curvature':
            wrapped_array = np.pad(row_data, pad_width=2, mode='wrap')
            curv_data = np.diff(wrapped_array, n=2)[1: -1]
            assert curv_data.shape == row_data.shape, "curv_data and row_data should be same shape."
            if max(np.abs(curv_data)) >= threshold:
                return 3
        elif test_type == 'length':
            row_data_plus_one = np.roll(row_data, 1)
            for indx in range(row_data.shape[0]):
                this_cell = row_data[indx]
                next_cell = row_data_plus_one[indx]
                delta_threshold = this_cell * threshold
                if abs(this_cell - next_cell) >= delta_threshold:
                    # edge extraction made a mistake on this frame, return 3
                    return 3
        else:
            raise Exception('You should not have gotten this far. Why isnt my assert working?')
    # if you get this far, edge extraction was probably fine, return 1
    return 1
