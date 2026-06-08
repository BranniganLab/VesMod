#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:09:33 2026

@author: js2746
"""

import numpy as np
import math
from scipy.special import lpmv
from scipy.Constants import Boltzmann
from lmfit import Model


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


def calc_tension_from_reduced_tension(r0, reduced_sigma, kc, temperature):
    """
    Convert a dimensionless reduced membrane tension to a physical tension.

    The reduced tension used in the Hackl, Seifert, and Sackmann (1997)
    fluctuation spectrum theory is related to the physical membrane tension
    by

    sigma = reduced_sigma * kc * k_B * T / r0^2

    where ``kc`` is expressed in units of kBT and ``r0`` is the vesicle radius.

    Parameters
    ----------
    r0 : float
        Average vesicle radius in microns.
    reduced_sigma : float
        Dimensionless reduced tension obtained from fitting the fluctuation
        spectrum.
    kc : float
        Membrane bending modulus in units of kBT.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Physical membrane tension in N/m (equivalently J/m²).

    Notes
    -----
    The input radius is converted from microns to meters before computing
    the tension.

    """
    one_kBT = kBT(temperature)
    r0_meter = r0 / 1e6
    r0_meter2 = r0_meter ** 2
    sigma = reduced_sigma * kc * one_kBT / r0_meter2  # units of J/m^2 or N/m (same thing)
    return sigma


def kBT(temperature: float) -> float:
    """
    Calculate the thermal energy k_B T.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Thermal energy k_B T in Joules.
    """
    k_B = Boltzmann  # J/K
    return k_B * temperature


def area_change_pct(sigma, ka):
    return (sigma / ka) * 100
