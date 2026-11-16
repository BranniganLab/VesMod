"""Relaxation times for spherical vesicle modes."""

import numpy as np
from scipy.constants import Boltzmann

from .spherical_validation import validate_spherical_inputs, validate_vesicle_radii


def mode_relaxation_times(ell, kC, sigma, *, config, radii):
    """Return spherical-mode relaxation times in seconds, shape (radii, ell).

    kC is in kBT, sigma is reduced tension, and radii are in microns.
    The explicitly selected relaxation model determines the drag expression.
    This calculation can also be used without enabling camera integration.

    Parameters
    ----------
    ell
        Iterable of spherical harmonic degrees.
    kC
        Bending modulus in kBT.
    sigma
        Dimensionless reduced tension.
    config
        Fit settings including the relaxation model, viscosities, and temperature.
    radii
        Replica radii in micrometers.

    Returns
    -------
    numpy.ndarray
        Relaxation times in seconds, indexed by radius and degree.
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
    radii_m = validate_vesicle_radii(radii) * 1e-6
    if sigma <= -6:
        raise ValueError("Relaxation requires stable spherical modes: sigma > -6.")
    drag_law, validate = RELAXATION_MODELS[config.relaxation_model]
    validate(config, require_viscosities=True)
    rate_scale = kC * Boltzmann * config.temperature / (config.viscosity_out * radii_m**3)
    drag = drag_law(ell, radii_m, config)
    return drag / (rate_scale[:, None] * stiffness * ell * (ell + 1))


def _solvent_drag(ell, _radii_m, config):
    """Drag with independent interior and exterior solvent viscosities.

    Parameters
    ----------
    ell
        Array of spherical harmonic degrees.
    _radii_m
        Replica radii in meters, unused by this law.
    config
        Fit settings including the relaxation model, viscosities, and temperature.

    Returns
    -------
    numpy.ndarray
        Dimensionless drag per degree, shared across radii.
    """
    return (4 * ell**3 + 6 * ell**2 - 1
            + (2 * ell**3 + 3 * ell**2 - 5)
            * (config.viscosity_in / config.viscosity_out - 1))


def _solvent_and_membrane_drag(ell, radii_m, config):
    """Drag with equal solvent viscosities and membrane surface viscosity.

    Parameters
    ----------
    ell
        Array of spherical harmonic degrees.
    radii_m
        Replica radii in meters.
    config
        Fit settings including the relaxation model, viscosities, and temperature.

    Returns
    -------
    numpy.ndarray
        Dimensionless drag indexed by radius and degree.
    """
    chi_s = config.eta_m / (radii_m[:, None] * config.viscosity_out)
    return (4 * ell**3 + 6 * ell**2 - 1) + (4 * ell**2 + 4 * ell - 8) * chi_s


def _validate_solvent(config, *, require_viscosities):
    """Validate solvent.

    Parameters
    ----------
    config
        Fit settings including the relaxation model, viscosities, and temperature.
    require_viscosities
        Whether the viscosities needed for evaluation must be present.

    Returns
    -------
    None
        Raises ValueError for incompatible settings.
    """
    if config.eta_m is not None:
        raise ValueError("eta_m requires relaxation_model='solvent_and_membrane'.")
    if require_viscosities and (config.viscosity_in is None or config.viscosity_out is None):
        raise ValueError("viscosity_in and viscosity_out are required for solvent_only relaxation.")


def _validate_solvent_and_membrane(config, *, require_viscosities):
    """Validate solvent and membrane.

    Parameters
    ----------
    config
        Fit settings including the relaxation model, viscosities, and temperature.
    require_viscosities
        Whether the viscosities needed for evaluation must be present.

    Returns
    -------
    None
        Raises ValueError for incompatible settings.
    """
    if config.eta_m is None:
        raise ValueError("eta_m is required for solvent_and_membrane (zero is allowed).")
    if (config.viscosity_in is not None and config.viscosity_out is not None
            and config.viscosity_in != config.viscosity_out):
        raise ValueError("Faizi 2024 Eq. 1 requires equal solvent viscosities; "
                         "omit viscosity_in or set it equal to viscosity_out.")
    if require_viscosities and config.viscosity_out is None:
        raise ValueError("viscosity_out is required for solvent_and_membrane relaxation.")


# Each entry pairs a spherical drag law with validation of its settings.
RELAXATION_MODELS = {
    "solvent_only": (_solvent_drag, _validate_solvent),
    "solvent_and_membrane": (_solvent_and_membrane_drag, _validate_solvent_and_membrane),
}
