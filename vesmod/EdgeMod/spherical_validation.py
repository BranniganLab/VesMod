"""Validate inputs shared by spherical power and relaxation calculations."""

import math
from numbers import Real

import numpy as np


def validate_spherical_inputs(q, kC, sigma, lmax) -> list[int]:
    """Validate the spherical spectrum domain and return integer modes."""
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
            raise ValueError("spherical_contour_spectrum requires q >= 2.")
        if mode_int > lmax_int:
            raise ValueError("Each q mode must be less than or equal to lmax.")
        integer_modes.append(mode_int)
    return integer_modes


def validate_vesicle_radii(radii):
    """Require positive radii in microns for fitting and camera averaging."""
    if radii is None:
        raise ValueError("Vesicle radii are required for spectrum fitting.")
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if radii.ndim != 1 or radii.size == 0 or np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("Spectrum fitting radii must be finite and positive.")
    return radii.copy()
