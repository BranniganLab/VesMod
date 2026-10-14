"""Built-in Fourier power calculators for uniformly sampled contours."""

from dataclasses import dataclass

import numpy as np

from .power_calculation import PowerCalculationResult


def _coefficients(radii):
    """Calculate the historical reference radius and normalized coefficients."""
    radius = float(np.mean(radii))
    coefficients = np.fft.fft(radii, axis=1, norm='backward') / (radius * radii.shape[1])
    return radius, coefficients


def _result(radius, coefficients, definition):
    """Package full two-sided powers in signed FFT ordering."""
    count = coefficients.shape[1]
    modes = np.rint(np.fft.fftfreq(count) * count).astype(int)
    return PowerCalculationResult(
        modes=modes, powers=np.mean(np.abs(coefficients) ** 2, axis=0),
        reference_radius=radius, radius_unit='um',
        power_definition=definition, power_unit='dimensionless',
    )


@dataclass(frozen=True)
class MeanSquaredPowerCalculator:
    """Calculate mean squared normalized complex Fourier amplitudes."""

    def __call__(self, radii: np.ndarray) -> PowerCalculationResult:
        """Return mean-square power, including persistent deformation."""
        radius, coefficients = _coefficients(radii)
        return _result(radius, coefficients, 'mean_square')


@dataclass(frozen=True)
class TemporalVarianceCalculator:
    """Calculate population variance around each complex mode's temporal mean."""

    def __call__(self, radii: np.ndarray) -> PowerCalculationResult:
        """Center complex coefficients before averaging squared magnitudes."""
        if radii.shape[0] < 2:
            raise ValueError('temporal_variance requires at least two frames.')
        radius, coefficients = _coefficients(radii)
        coefficients = coefficients - np.mean(coefficients, axis=0, keepdims=True)
        return _result(radius, coefficients, 'temporal_variance')
