"""Estimate membrane parameters from measured contour powers."""

import logging
from dataclasses import replace

from lmfit import Model
import numpy as np

from .spherical_model import configure_spectrum_model, spherical_contour_spectrum

from .range_selection.selection import select_manual_range, slice_fit_arrays

logger = logging.getLogger(__name__)


def _log_relative_rmse(result, powers):
    """Report residual magnitude without applying fit acceptance criteria.

    Parameters
    ----------
    result : lmfit.model.ModelResult
        Optimizer result containing the best-fit powers.
    powers : array-like
        Observed powers aligned with the fit.

    Returns
    -------
    None
        Logs the relative RMSE.
    """
    measured = np.asarray(powers)
    residuals = measured - np.asarray(result.best_fit)
    rmse = np.sqrt(np.mean(residuals**2))
    mean_power = np.mean(np.abs(measured))
    relative_rmse = (rmse / mean_power if mean_power > 0
                     else (np.nan if rmse == 0 else np.inf))
    logger.info("Spectrum fit relative RMSE=%g", relative_rmse)


def fit_spectrum(modes, powers, lmax, free_sigma=False, weighted=False,
                       *, kc_bounds, sigma_bounds, sem=None, model=spherical_contour_spectrum):
    """Fit selected modes and powers, with optional per-mode replica SEM.

    The model takes q, kC, sigma, and lmax and returns powers at q.

    Supply (minimum, maximum) pairs for kC and reduced tension.

    Parameters
    ----------
    modes : array-like
        Array of selected Fourier modes.
    powers : array-like
        Measured powers aligned with modes.
    lmax : int
        Inclusive model summation limit.
    free_sigma : bool
        Whether to vary reduced tension.
    weighted : bool
        Whether to weight by inverse replica SEM.
    kc_bounds : tuple of float
        Minimum and maximum bending modulus in kBT.
    sigma_bounds : tuple of float
        Minimum and maximum reduced tension.
    sem : array-like or None
        Optional per-mode replica standard errors.
    model : callable
        Power predictor called with q, kC, sigma, and lmax.

    Returns
    -------
    lmfit.model.ModelResult
        Optimizer result and fit diagnostics.
    """
    lmfit_model = Model(model)
    pars = lmfit_model.make_params(
        kC={"value": 15, "min": kc_bounds[0], "max": kc_bounds[1], "vary": True},
        sigma={"value": 0, "min": sigma_bounds[0], "max": sigma_bounds[1], "vary": free_sigma},
        lmax={"value": lmax, "vary": False},
    )

    fit_kwargs = {"q": modes, "params": pars, "max_nfev": 20000}
    if weighted:
        if sem is None:
            raise ValueError("Replica SEM values are required for a weighted fit.")
        uncertainties = np.asarray(sem, dtype=float)
        amplitudes = np.asarray(powers, dtype=float)
        if uncertainties.shape != amplitudes.shape:
            raise ValueError("Replica SEM values must match the fitted amplitudes.")
        if not np.all(np.isfinite(uncertainties)) or np.any(uncertainties <= 0):
            raise ValueError(
                "Replica SEM values must be finite and greater than zero "
                "for every fitted mode."
            )
        fit_kwargs["weights"] = 1 / uncertainties

    result = lmfit_model.fit(powers, **fit_kwargs)
    _log_relative_rmse(result, powers)
    return result


def fit_spectrum_to_theory(
    modes, powers, lmax, free_sigma=False, weighted=False, *, sem=None, radii
):
    """Fit selected contour modes and return bending modulus and reduced tension.

    Parameters
    ----------
    modes : array-like
        Array of selected Fourier modes.
    powers : array-like
        Measured powers aligned with modes.
    lmax : int
        Inclusive model summation limit.
    free_sigma : bool
        Whether to vary reduced tension.
    weighted : bool
        Whether to weight by inverse replica SEM.
    sem : array-like or None
        Optional per-mode replica standard errors.
    radii : float or array-like
        Replica radii in micrometers.

    Returns
    -------
    tuple
        Fitted bending modulus in kBT and dimensionless reduced tension.
    """
    model, kc_bounds, sigma_bounds = configure_spectrum_model(None, radii)
    result = fit_spectrum(
        modes, powers, lmax, free_sigma, weighted, sem=sem,
        model=model, kc_bounds=kc_bounds, sigma_bounds=sigma_bounds,
    )
    return result.best_values["kC"], result.best_values["sigma"]

def fit_configured_spectrum(modes, powers, config, radii, *, model=None, sem=None,
                            weighted=False, expected_replica_count=None):
    """Select modes and fit using a factory that supplies a predictor and bounds.

    Parameters
    ----------
    modes : numpy.ndarray
        Available Fourier modes.
    powers : array-like
        Powers aligned with modes.
    config : SpectrumFitConfig
        Fit settings including mode bounds.
    radii : array-like
        Replica radii in micrometers.
    model : callable or None
        Optional factory returning a predictor and two parameter bound pairs.
    sem : array-like or None
        Optional standard errors aligned with modes.
    weighted : bool
        Whether to weight by inverse SEM.
    expected_replica_count : int or None
        Optional number of expected radii.

    Returns
    -------
    tuple
        Optimizer result and effective fit configuration.
    """
    if model is None:
        if config.model != "spherical_contour_spectrum":
            raise ValueError(f"Unknown model {config.model!r}; supply a model factory.")
        model = configure_spectrum_model
    elif config.model == "spherical_contour_spectrum":
        config = replace(config, model=getattr(model, "__name__", type(model).__name__))
    fit_mode_mask = select_manual_range(modes, config.lower_bound, config.upper_bound)
    modes, powers, sem = slice_fit_arrays(fit_mode_mask, modes, powers, sem=sem)
    if modes.size == 0:
        raise ValueError("Configured q range contains no spectrum modes.")
    predict, kc_bounds, sigma_bounds = model(
        config, radii, expected_replica_count=expected_replica_count
    )
    result = fit_spectrum(
        modes, powers, config.lmax, config.free_sigma, weighted=weighted,
        sem=sem, model=predict, kc_bounds=kc_bounds, sigma_bounds=sigma_bounds,
    )
    if config.camera_integration:
        instantaneous_config = replace(config, camera_integration=False)
        fitted_radii = tuple(radii)

        def evaluate_instantaneous(q):
            """Evaluate instantaneous.

            Parameters
            ----------
            q : array-like
                Contour Fourier modes to evaluate.

            Returns
            -------
            array-like
                Instantaneous powers at the fitted parameters.
            """
            predictor, _, _ = model(
                instantaneous_config, list(fitted_radii),
                expected_replica_count=expected_replica_count,
            )
            return predictor(
                q, result.best_values["kC"], result.best_values["sigma"], config.lmax
            )

        result.eval_instantaneous = evaluate_instantaneous
    return result, config
