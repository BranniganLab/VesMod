"""Estimate membrane parameters from measured contour powers."""

import logging

from lmfit import Model
import numpy as np

from .spherical_model import configure_spectrum_model, spherical_contour_spectrum

logger = logging.getLogger(__name__)


def _log_relative_rmse(result, powers):
    """Report residual magnitude without applying fit acceptance criteria."""
    measured = np.asarray(powers)
    residuals = measured - np.asarray(result.best_fit)
    rmse = np.sqrt(np.mean(residuals**2))
    mean_power = np.mean(np.abs(measured))
    relative_rmse = (rmse / mean_power if mean_power > 0
                     else (np.nan if rmse == 0 else np.inf))
    logger.info("Spectrum fit relative RMSE=%g", relative_rmse)


def fit_spectrum(modes, powers, lmax, free_sigma=False, weighted=False,
                       *, kc_bounds, sigma_bounds, sem=None, predict=spherical_contour_spectrum):
    """Fit selected modes and powers, with optional per-mode replica SEM.

    The predictor takes q, kC, sigma, and lmax and returns powers at q.

    Supply (minimum, maximum) pairs for kC and reduced tension.
    """
    model = Model(predict)
    pars = model.make_params(
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

    result = model.fit(powers, **fit_kwargs)
    _log_relative_rmse(result, powers)
    return result


def fit_spectrum_to_theory(
    modes, powers, lmax, free_sigma=False, weighted=False, *, sem=None, radii
):
    """Fit selected contour modes and return bending modulus and reduced tension."""
    predict, kc_bounds, sigma_bounds = configure_spectrum_model(None, radii)
    result = fit_spectrum(
        modes, powers, lmax, free_sigma, weighted, sem=sem,
        predict=predict, kc_bounds=kc_bounds, sigma_bounds=sigma_bounds,
    )
    return result.best_values["kC"], result.best_values["sigma"]
