"""Shared geometry helpers for radial contour processing."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RadialBaselineFit:
    """Low-order Fourier representation of a uniformly sampled contour."""
    values: np.ndarray
    order: int


def fit_radial_baseline(radii, order):
    """Fit a low-order Fourier baseline to uniformly sampled radial values."""
    if isinstance(radii, list):
        radii = np.asarray(radii)
    if not isinstance(radii, np.ndarray) or radii.ndim != 1:
        raise TypeError("radii must be a 1D numpy array or list")
    if not isinstance(order, int):
        raise TypeError("order must be an int")
    if order < 0:
        raise ValueError("order must be a positive integer")
    if order >= radii.shape[0] // 2:
        raise IndexError(f"radii does not have enough modes ({radii.shape[0]}) to retain the lowest {order}.")
    spectrum = np.fft.fft(radii)
    if order == 0:
        spectrum[1:] = 0
    else:
        spectrum[order + 1 : -order] = 0
    return RadialBaselineFit(np.fft.ifft(spectrum).real, order)
