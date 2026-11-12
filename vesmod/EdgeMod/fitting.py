"""Estimate membrane parameters from measured contour powers."""

import logging

from lmfit import Model
import numpy as np

from .spherical_model import spherical_contour_spectrum, validate_vesicle_radii

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
                       *, sem=None, radii=None, expected_replica_count=None,
                       predict=spherical_contour_spectrum, sigma_min=-100):
    """Fit selected modes and powers, with optional per-mode replica SEM.

    The predictor takes q, kC, sigma, and lmax and returns powers at q.

    Radii in microns are required for every fit. The reduced-tension upper
    bound is 100 * mean(radii)**2. For ensembles, the arithmetic mean of
    the replica mean radii sets the bound on shared reduced tension.

    Supply ``expected_replica_count`` when fitting
    an averaged spectrum directly. This checks that there is one radius per
    contributing replica; an averaged spectrum alone cannot reveal its count.
    ``Spectrum`` and ``SpectrumEnsemble`` supply the count automatically.
    """
    radii = validate_vesicle_radii(radii)
    sigma_max = 100 * float(np.mean(radii))**2
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

    model = Model(predict)
    pars = model.make_params(
        kC={"value": 15, "min": 1, "max": 500, "vary": True},
        sigma={"value": 0, "min": sigma_min, "max": sigma_max, "vary": free_sigma},
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
    result = fit_spectrum(
        modes, powers, lmax, free_sigma, weighted, sem=sem, radii=radii
    )
    return result.best_values["kC"], result.best_values["sigma"]
