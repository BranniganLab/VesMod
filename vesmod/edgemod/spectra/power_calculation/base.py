"""Contracts for replaceable contour-to-spectrum power calculations."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vesmod.validation import require_positive_real



@dataclass(frozen=True)
class PowerCalculationResult:
    """Calculated mode powers and their scientific interpretation.

    Parameters
    ----------
    modes, powers : array-like
        Unique signed integer modes and matching finite nonnegative powers.
    reference_radius : float
        Positive reference radius used by the calculation.
    radius_unit, power_definition, power_unit : str
        Explicit units and description of the calculated observable.
    """

    modes: np.ndarray
    powers: np.ndarray
    reference_radius: float
    radius_unit: str
    power_definition: str
    power_unit: str

    def __post_init__(self):
        """Validate mode powers and scientific metadata."""
        modes = np.asarray(self.modes)
        powers = np.asarray(self.powers)
        if modes.ndim != 1 or powers.ndim != 1 or not modes.size:
            raise ValueError("modes and powers must be nonempty one-dimensional arrays.")
        if modes.shape != powers.shape:
            raise ValueError("modes and powers must have matching lengths.")
        if modes.dtype.kind not in 'iu' or np.unique(modes).size != modes.size:
            raise ValueError("modes must be unique integers.")
        if powers.dtype.kind not in 'fiu' or not np.all(np.isfinite(powers)):
            raise ValueError("powers must contain finite real numbers.")
        if np.any(powers < 0):
            raise ValueError("powers must be nonnegative.")
        for name in ('radius_unit', 'power_definition', 'power_unit'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string.")
        object.__setattr__(self, 'reference_radius',
                           require_positive_real(self.reference_radius, 'reference_radius'))
        object.__setattr__(self, 'modes', modes.copy())
        object.__setattr__(self, 'powers', powers.astype(float, copy=True))


class SpectrumCalculator(Protocol):
    """A callable converting validated radii in micrometers to powers."""

    def __call__(self, radii: np.ndarray) -> PowerCalculationResult:
        """Calculate powers without mutating the input trajectory."""
