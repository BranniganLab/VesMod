"""Relaxation times for spherical vesicle modes."""

import numpy as np
from scipy.constants import Boltzmann

from .spherical_validation import validate_spherical_inputs, validate_vesicle_radii


def mode_relaxation_times(ell, kC, sigma, *, config, radii):
    """Return spherical-mode relaxation times in seconds, shape (radii, ell).

    kC is in kBT, sigma is reduced tension, and radii are in microns.
    The explicitly selected relaxation model determines the drag expression.
    This calculation can also be used without enabling camera integration.
    """
    try:
        modes = list(ell)
    except TypeError as error:
        raise TypeError("ell must be an iterable of integer modes.") from error
    ell = np.asarray(
        validate_spherical_inputs(modes, kC, sigma, max(modes, default=2)),
        dtype=float,
    )
    stiffness = (ell - 1) * (ell + 2) * (ell * (ell + 1) + sigma)
    calculate_times = prepare_spherical_relaxation(kC, sigma, config=config, radii=radii)
    return calculate_times(ell, stiffness)


def prepare_spherical_relaxation(kC, sigma, *, config, radii):
    """Bind dynamics for one trial parameter set and its replica radii."""
    radii_m = validate_vesicle_radii(radii) * 1e-6
    _validate_relaxation_inputs(sigma, config)
    rate_scale = kC * Boltzmann * config.temperature / (config.viscosity_out * radii_m**3)

    def calculate_times(ell, stiffness):
        return _mode_relaxation_times_unchecked(ell, stiffness, config, radii_m, rate_scale)

    return calculate_times


def _validate_relaxation_inputs(sigma, config):
    """Check stability and solvent inputs once at the public API boundary."""
    if sigma <= -6:
        raise ValueError("Relaxation requires stable spherical modes: sigma > -6.")
    if config.viscosity_out is None:
        raise ValueError("viscosity_out is required to calculate relaxation times.")
    if config.relaxation_model == "solvent_only" and config.viscosity_in is None:
        raise ValueError("viscosity_in is required for solvent_only relaxation times.")


def _mode_relaxation_times_unchecked(ell, stiffness, config, radii_m, rate_scale):
    """Evaluate dynamics for validated modes and precomputed SI radius terms.

    The prediction loop uses cached spherical-mode arrays, so it must not
    call the public mode_relaxation_times API and revalidate every ell.
    The public API and predictor share this numerical calculation.
    """
    drag = 4 * ell**3 + 6 * ell**2 - 1
    if config.relaxation_model == "solvent_only":
        drag += ((2 * ell**3 + 3 * ell**2 - 5)
                 * (config.viscosity_in / config.viscosity_out - 1))
    else:
        chi_s = config.eta_m / (radii_m[:, None] * config.viscosity_out)
        drag = drag[None, :] + (4 * ell**2 + 4 * ell - 8) * chi_s
    return drag / (rate_scale[:, None] * stiffness * ell * (ell + 1))
