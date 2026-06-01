#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:09:33 2026

@author: js2746
"""

import numpy as np
import math
from scipy.special import lpmv
from lmfit import Model


def downsample_to_new_indices(data: np.ndarray, index_floats: np.ndarray) -> np.ndarray:
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


def fit_spectrum_to_theory_lmfit(fitting_group, lmax, free_sigma=False, weighted=False):
    """
    Fit a Mini_spectrum to the theory from Hackl, Seifert, and Sachmann 1997 \
    using lmfit.

    Parameters
    ----------
    fitting_group : namedtuple
        Mini_spectrum containing modes, avg_amps2, and std_amps2 of just the \
        modes you wish to fit to.
    lmax : int
        Upper bound on the summation. Usually = the highest positive mode \
        present in the full spectrum.

    Returns
    -------
    kC : float
        The kC, extracted from the fit of fitting_group to the theoretical \
        expression.
    pcov : ndarray
        The covariance matrix.

    """
    model = Model(HSS97)
    pars = model.make_params(kC={'value': 15, 'min': 1, 'max': 500, 'vary': True}, sigma={'value': 0, 'min': -100, 'max': 1000, 'vary': free_sigma}, lmax={'value': lmax, 'vary': False})
    if weighted and fitting_group.std_amps2.all():
        result = model.fit(fitting_group.avg_amps2, q=fitting_group.modes, weights=(1 / fitting_group.std_amps2), params=pars, max_nfev=20000)
    else:
        result = model.fit(fitting_group.avg_amps2, q=fitting_group.modes, params=pars, max_nfev=20000)
    if result.best_values['kC'] != 15:
        return result.best_values['kC'], result.best_values['sigma']
    print(f"kC: {result.best_values['kC']} and sigma: {result.best_values['sigma']}")
    raise ValueError("Fitting did not converge. Best fit for kC equals initial guess (15 kBT).")


def HSS97(q, kC, sigma, lmax):
    """
    Generate function to be fit in order to estimate kC and sigma. See Hackl,\
    Seifert, and Sackmann 1997 eqs 7 & 8.

    Parameters
    ----------
    q : int
        Wave number / independent variable.
    kC : float
        Bending modulus to be fit.
    sigma : float
        Effective tension to be fit. 
    lmax : int
        Maximum value to sum up to.

    Returns
    -------
    function : list
        The values to be fit.

    """
    function = []
    lmax = int(lmax)
    for wavenum in q:
        wavenum = int(wavenum)
        summ = 0
        for l in range(wavenum, lmax):
            denom = (l - 1) * (l + 2) * (l ** 2 + l + sigma)
            summ += Nlq_Plq0_squared(l, wavenum) / denom
        function.append((1 / kC) * summ)
    return function


def Nlq_Plq0_squared(l, q):
    """
    Generate the nomarlized associated legendre polynomial N_{lq} * P_{lq}(0)\
    squared. See Hackl, Seifert, and Sackmann 1997 for more details.

    Parameters
    ----------
    l : int
        Polynomial order.
    q : int
        Polynomial degree / wave number.

    Returns
    -------
    plq : float
        The associated legendre polynomial evaluated at zero.

    """
    assert q <= l, "q must be <= l"
    assert q >= 0, "q must be non-negative for lpmv to work"
    Nlq = (2 * l + 1) / (4 * np.pi)
    Nlq *= (math.factorial(l - q) / math.factorial(l + q))
    sqrtNlq = np.sqrt(Nlq)
    Plq0 = lpmv(q, l, 0)
    np.seterr(all='raise')
    try:
        total = (sqrtNlq * Plq0)**2
    except FloatingPointError:
        raise Exception(f"FloatingPointError when l = {l}; q = {q}")
    return total


def calc_sigma_from_reduced_sigma(r0, reduced_sigma, kc):
    kBT_295 = 4.0728e-21
    r0_micron = r0 / 13.44  # comes from Josh's microscope settings; 13.44 pixels = 1 micron
    r0_meter = r0_micron / 1e6
    r0_meter2 = r0_meter ** 2
    sigma = reduced_sigma * kc * kBT_295 / r0_meter2  # units of J/m^2 or N/m (same thing)
    return sigma


def area_change_pct(sigma, ka):
    return (sigma / ka) * 100
