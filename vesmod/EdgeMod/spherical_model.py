"""Spherical model for vesicle contour fluctuation spectra."""

from functools import lru_cache
from numbers import Integral
import numpy as np
from scipy.constants import Boltzmann
from scipy.special import gammaln

from .camera_integration import exposure_power_factor
from .relaxation import prepare_spherical_relaxation
from .spherical_validation import validate_spherical_inputs, validate_vesicle_radii


def spherical_contour_spectrum(q: list[int], kC: float, sigma: float, lmax: int) -> list[float]:
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
    modes = validate_spherical_inputs(q, kC, sigma, lmax)
    function = []
    lmax = int(lmax)
    for wavenum in modes:
        summ = 0.0
        for l in range(wavenum, lmax + 1):
            denom = (l - 1) * (l + 2) * (l ** 2 + l + sigma)
            if denom == 0:
                raise ValueError(
                    "spherical_contour_spectrum denominator is zero for the requested sigma and mode."
                )
            summ += Nlq_Plq0_squared(l, wavenum) / denom
        function.append((1 / kC) * summ)
    return function


@lru_cache(maxsize=256)
def _camera_projection(q, lmax):
    """Cache fixed Nlq Plq^2 geometry for repeated optimizer evaluations."""
    ell = np.arange(q, lmax + 1, 2, dtype=float)
    weights = np.array([Nlq_Plq0_squared(int(l), q) for l in ell])
    return ell, weights


def predict_spectrum(
    q, kC, sigma, lmax, *, config, radii=None, relaxation=prepare_spherical_relaxation
):
    """Predict dimensionless complex contour power for fitting or plotting.

    The equilibrium model, camera-integration toggle, and relaxation model
    are explicit independent settings. Exposure averaging multiplies each
    spherical-mode power by B(t_exp/tau_l) before contour projection. Ensemble
    predictions average equally across replica radii, with shared kC and
    reduced tension. Disabling camera integration needs no radii or solvent
    viscosities and returns the equilibrium spectrum exactly.
    """
    modes = validate_spherical_inputs(q, kC, sigma, lmax)
    if not config.camera_integration:
        return spherical_contour_spectrum(modes, kC, sigma, lmax)
    calculate_times = relaxation(kC, sigma, config=config, radii=radii)
    predictions = []
    for mode in modes:
        ell, weights = _camera_projection(mode, int(lmax))
        stiffness = (ell - 1) * (ell + 2) * (ell * (ell + 1) + sigma)
        times = calculate_times(ell, stiffness)
        retained = exposure_power_factor(config.exposure_time / times).mean(axis=0)
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
        The squared normalized associated Legendre value for contour projection.

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



def configure_spectrum_model(
    config, radii, *, expected_replica_count=None, relaxation=prepare_spherical_relaxation
):
    """Return a predictor and the bending-modulus and reduced-tension bounds."""
    radii = validate_vesicle_radii(radii)
    if expected_replica_count is not None:
        if (isinstance(expected_replica_count, (bool, np.bool_))
                or not isinstance(expected_replica_count, (int, np.integer))
                or expected_replica_count < 1):
            raise ValueError("expected_replica_count must be a positive integer.")
        if radii.size != expected_replica_count:
            raise ValueError(
                "Spectrum fitting requires one radius per replica: "
                f"expected {expected_replica_count} radii, received {radii.size}. "
                "Supply each replica's radius, rather than a mean radius."
            )

    if config is None:
        predict = spherical_contour_spectrum
    else:
        def predict(q, kC, sigma, lmax):
            return predict_spectrum(
                q, kC, sigma, lmax, config=config, radii=radii, relaxation=relaxation
            )

    # Exposure averaging requires stability of the spherical l=2 mode.
    sigma_min = -6 + 1e-8 if config is not None and config.camera_integration else -100
    sigma_max = 100 * float(np.mean(radii))**2
    return predict, (1, 500), (sigma_min, sigma_max)
