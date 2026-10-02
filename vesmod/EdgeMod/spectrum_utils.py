#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:09:33 2026

@author: js2746
"""
from functools import lru_cache
from collections import namedtuple
from numbers import Integral, Real
import math
import numpy as np
from scipy.constants import Boltzmann
from scipy.special import gammaln
from lmfit import Model

MiniSpectrum = namedtuple("MiniSpectrum", ['modes', 'avg_amps2', 'avg_amps2_ste'])


def validate_lmfit_result(result, fitting_group, free_sigma):
    """Raise ValueError if the lmfit result is not physically or numerically reliable."""
    if not result.success:
        raise ValueError(f"Spectrum fit failed: {result.message}")

    kC = result.params["kC"]
    sigma = result.params["sigma"]

    if kC.value <= kC.min or kC.value >= kC.max:
        raise ValueError(f"Spectrum fit put kC on a parameter bound: kC={kC.value}")

    if free_sigma and (sigma.value <= sigma.min or sigma.value >= sigma.max):
        raise ValueError(
            f"Spectrum fit put sigma on a parameter bound: sigma={sigma.value}"
        )

    if kC.stderr is None:
        raise ValueError("Spectrum fit did not estimate uncertainty for kC.")

    if kC.stderr / abs(kC.value) > 0.5:
        raise ValueError(
            f"Spectrum fit has poorly constrained kC: "
            f"kC={kC.value}, stderr={kC.stderr}"
        )

    if free_sigma:
        correlation = (kC.correl or {}).get("sigma")
        if correlation is None or not np.isfinite(correlation):
            raise ValueError("Spectrum fit did not estimate a finite kC-sigma correlation.")
        if abs(correlation) > 0.95:
            raise ValueError(
                "Spectrum fit has strongly correlated kC and sigma: "
                f"correlation={correlation:.6g}, maximum absolute correlation=0.95"
            )



def fit_spectrum_lmfit(fitting_group, lmax, free_sigma=False, weighted=False,
                       *, config=None, radii=None):
    """Return the complete lmfit result for a theoretical spectrum fit."""
    model_function = HSS97
    if config is not None and config.exposure_time > 0:
        radii = _validate_camera_radii(radii)

        def camera_model(q, kC, sigma, lmax):
            return HSS97_with_camera_integration_time(q, kC, sigma, lmax, config=config, radii=radii)

        model_function = camera_model

    model = Model(model_function)
    pars = model.make_params(kC={'value': 15, 'min': 1, 'max': 500, 'vary': True}, sigma={'value': 0, 'min': -100, 'max': 1000, 'vary': free_sigma}, lmax={'value': lmax, 'vary': False})

    if config is not None and config.exposure_time > 0:
        # All spherical modes must be stable, including l=2 even if q=2
        # is omitted from the fitted contour spectrum.
        pars["sigma"].set(min=-6 + 1e-8)

    fit_kwargs = {"q": fitting_group.modes, "params": pars, "max_nfev": 20000}
    if weighted:
        uncertainties = fitting_group.avg_amps2_ste
        if uncertainties is None:
            raise ValueError("Replica SEM values are required for a weighted fit.")
        uncertainties = np.asarray(uncertainties, dtype=float)
        amplitudes = np.asarray(fitting_group.avg_amps2, dtype=float)
        if uncertainties.shape != amplitudes.shape:
            raise ValueError("Replica SEM values must match the fitted amplitudes.")
        if not np.all(np.isfinite(uncertainties)) or np.any(uncertainties <= 0):
            raise ValueError(
                "Replica SEM values must be finite and greater than zero "
                "for every fitted mode."
            )
        fit_kwargs["weights"] = 1 / uncertainties

    return model.fit(fitting_group.avg_amps2, **fit_kwargs)


def fit_spectrum_to_theory_lmfit(fitting_group, lmax, free_sigma=False, weighted=False):
    """
    Fit a Mini_spectrum to the theory from Hackl, Seifert, and Sachmann 1997 \
    using lmfit.

    Parameters
    ----------
    fitting_group : namedtuple
        Mini_spectrum containing modes, avg_amps2, and avg_amps2_ste of just the \
        modes you wish to fit to.
    lmax : int
        Inclusive upper bound on the summation.

    Returns
    -------
    kC : float
        The kC, extracted from the fit of fitting_group to the theoretical \
        expression.
    pcov : ndarray
        The covariance matrix.

    """
    result = fit_spectrum_lmfit(
        fitting_group,
        lmax,
        free_sigma,
        weighted,
    )
    validate_lmfit_result(result, fitting_group, free_sigma)

    return result.best_values['kC'], result.best_values['sigma']


def _validate_hss97_inputs(q, kC, sigma, lmax) -> list[int]:
    """Validate the public HSS97 domain and return integer modes."""
    if not isinstance(kC, Real) or isinstance(kC, bool):
        raise TypeError("kC must be numeric.")
    if not math.isfinite(kC) or kC <= 0:
        raise ValueError("kC must be finite and positive.")
    if not isinstance(sigma, Real) or isinstance(sigma, bool):
        raise TypeError("sigma must be numeric.")
    if not math.isfinite(sigma):
        raise ValueError("sigma must be finite.")
    if not isinstance(lmax, Real) or isinstance(lmax, bool):
        raise TypeError("lmax must be an integer-valued number.")
    if not math.isfinite(lmax) or not float(lmax).is_integer():
        raise ValueError("lmax must be a finite integer-valued number.")
    lmax_int = int(lmax)
    if lmax_int < 2:
        raise ValueError("lmax must be at least 2.")

    try:
        modes = list(q)
    except TypeError as error:
        raise TypeError("q must be an iterable of integer modes.") from error
    if not modes:
        raise ValueError("q must contain at least one mode.")

    integer_modes = []
    for mode in modes:
        # lmfit coerces independent variables to floating arrays, so accept
        # finite integer-valued Reals as well as Python/NumPy integer scalars.
        if not isinstance(mode, Real) or isinstance(mode, bool):
            raise TypeError("q must contain only integer-valued modes.")
        if not math.isfinite(mode) or not float(mode).is_integer():
            raise ValueError("q must contain only finite integer-valued modes.")
        mode_int = int(mode)
        if mode_int < 2:
            raise ValueError("HSS97 requires q >= 2.")
        if mode_int > lmax_int:
            raise ValueError("Each q mode must be less than or equal to lmax.")
        integer_modes.append(mode_int)
    return integer_modes


def HSS97(q: list[int], kC: float, sigma: float, lmax: int) -> list[float]:
    """
    Generate function to be fit in order to estimate kC and sigma. See Hackl,\
    Seifert, and Sackmann 1997 eqs 7 & 8.

    Parameters
    ----------
    q : list[int]
        Wave number / independent variable.
    kC : float
        Bending modulus to be fit.
    sigma : float
        Effective tension to be fit.
    lmax : int
        Inclusive upper value of the summation.

    Returns
    -------
    function : list
        The values to be fit.

    Notes
    -----
    Public calls require physical Fourier modes q >= 2 and q <= lmax.

    """
    modes = _validate_hss97_inputs(q, kC, sigma, lmax)
    function = []
    lmax = int(lmax)
    for wavenum in modes:
        summ = 0.0
        for l in range(wavenum, lmax + 1):
            denom = (l - 1) * (l + 2) * (l ** 2 + l + sigma)
            if denom == 0:
                raise ValueError(
                    "HSS97 denominator is zero for the requested sigma and mode."
                )
            summ += Nlq_Plq0_squared(l, wavenum) / denom
        function.append((1 / kC) * summ)
    return function


def exposure_power_factor(x):
    """Return retained power for a boxcar-averaged exponential correlation.

    ``x`` is exposure time / relaxation time. A series avoids cancellation
    near zero; the reciprocal form avoids squaring large x.
    """
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("Exposure/relaxation ratio must be finite and nonnegative.")
    result = np.empty_like(x)
    small = x < 1e-3
    z = x[small]
    result[small] = 1 - z / 3 + z**2 / 12 - z**3 / 60 + z**4 / 360
    z = x[~small]
    result[~small] = 2 / z * (1 + np.expm1(-z) / z)
    return result


def _validate_camera_radii(radii):
    """Require one positive radius in microns for each modeled replica."""
    if radii is None:
        raise ValueError("Vesicle radii are required for camera integration fitting.")
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if radii.ndim != 1 or radii.size == 0 or np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("Camera integration radii must be finite and positive.")
    return radii.copy()


@lru_cache(maxsize=256)
def _camera_projection(q, lmax):
    """Cache fixed Nlq Plq^2 geometry for repeated optimizer evaluations."""
    ell = np.arange(q, lmax + 1, 2, dtype=float)
    weights = np.array([Nlq_Plq0_squared(int(l), q) for l in ell])
    return ell, weights


def HSS97_with_camera_integration_time(q, kC, sigma, lmax, *, config, radii):
    """Predict exposure-averaged dimensionless complex contour power.

    Uses Faizi et al. (2020), eq. 2, solvent-only spherical relaxation rates.
    kC is in kBT, sigma is reduced tension, radii are in microns, exposure
    is in seconds, viscosities are in Pa s, and temperature is in Kelvin.
    Each l contribution is multiplied by B(t_exp/tau_l) before summation.
    For ensembles the predictions are averaged equally across replica radii
    with shared kC and reduced tension, matching the measured spectrum mean.
    """
    modes = _validate_hss97_inputs(q, kC, sigma, lmax)
    if config.exposure_time == 0:
        return HSS97(modes, kC, sigma, lmax)
    radii_m = _validate_camera_radii(radii) * 1e-6
    if sigma <= -6:
        raise ValueError("Camera integration requires stable spherical modes: sigma > -6.")
    rate_scale = kC * Boltzmann * config.temperature / (config.viscosity_out * radii_m**3)
    predictions = []
    for mode in modes:
        ell, weights = _camera_projection(mode, int(lmax))
        stiffness = (ell - 1) * (ell + 2) * (ell * (ell + 1) + sigma)
        drag = (4 * ell**3 + 6 * ell**2 - 1
                + (2 * ell**3 + 3 * ell**2 - 5)
                * (config.viscosity_in / config.viscosity_out - 1))
        rates = rate_scale[:, None] * (stiffness * ell * (ell + 1) / drag)
        retained = exposure_power_factor(config.exposure_time * rates).mean(axis=0)
        predictions.append(float(np.sum(weights * retained / stiffness) / kC))
    return predictions


def Nlq_Plq0_squared(l: int, q: int) -> float:
    """Return the squared normalized associated Legendre value at zero.

    The direct product of the spherical-harmonic normalization and
    scipy.special.lpmv is numerically unstable at high l and q: the
    normalization can underflow while the polynomial overflows. This
    implementation combines their analytic factorial expressions in log space
    so only the finite final value is exponentiated.

    Parameters
    ----------
    l : int
        Polynomial order.
    q : int
        Polynomial degree / wave number.

    Returns
    -------
    float
        The squared normalized associated Legendre value from HSS97.

    """
    for name, value in (("l", l), ("q", q)):
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
    if q < 0:
        raise ValueError("q must be non-negative.")
    if l < 0:
        raise ValueError("l must be non-negative.")
    if q > l:
        raise ValueError("q must be <= l.")

    l = int(l)
    q = int(q)
    if (l + q) % 2:
        return 0.0

    half_difference = (l - q) // 2
    half_sum = (l + q) // 2
    log_total = (
        np.log(2 * l + 1)
        - np.log(4 * np.pi)
        + gammaln(l - q + 1)
        + gammaln(l + q + 1)
        - 2 * l * np.log(2)
        - 2 * gammaln(half_difference + 1)
        - 2 * gammaln(half_sum + 1)
    )
    return float(np.exp(log_total))


def calc_tension_from_reduced_tension(
    r0: float,
    reduced_tension: float,
    kc: float,
    temperature: float,
) -> float:
    """
    Convert a dimensionless reduced membrane tension to a physical tension in N/m.

    The reduced tension used in the Hackl, Seifert, and Sackmann (1997)
    fluctuation spectrum theory is related to the physical membrane tension
    by

    sigma = tilde_sigma * kc * k_B * T / r0^2

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
    one_kBT = Boltzmann * temperature                   # units of Joules
    r0_meter = r0 / 1e6                                 # units of meters
    r0_meter2 = r0_meter ** 2                           # units of meters^2
    sigma = reduced_tension * kc * one_kBT / r0_meter2  # units of J/m^2 or N/m
    return sigma
