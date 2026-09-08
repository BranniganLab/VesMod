"""Shared geometry helpers for uniformly sampled radial contours.

The functions in this module operate on contours represented as radii sampled
at evenly spaced angles from zero through (but not including) 2π. They are
deliberately independent of extraction, QC, and persistence so the same
geometric projection can be reused by each layer.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RadialBaselineFit:
    """Low-order Fourier representation of a uniformly sampled contour."""
    values: np.ndarray
    order: int


def fit_radial_baseline(radii, order):
    """Fit a low-order Fourier baseline to uniformly sampled radial values.

    Parameters
    ----------
    radii : numpy.ndarray or list
        One-dimensional radial values sampled at evenly spaced angles.
    order : int
        Highest positive Fourier harmonic to retain. ``order=0`` retains only
        the mean radius.

    Returns
    -------
    RadialBaselineFit
        The reconstructed contour and the requested harmonic order.

    Raises
    ------
    TypeError
        If ``radii`` is not one-dimensional or ``order`` is not an integer.
    ValueError
        If ``order`` is negative.
    IndexError
        If the requested order is not supported by the number of samples.
    """
    if isinstance(radii, list):
        radii = np.asarray(radii)
    if not isinstance(radii, np.ndarray) or radii.ndim != 1:
        raise TypeError("radii must be a 1D numpy array or list")
    if not isinstance(order, int):
        raise TypeError("order must be an int")
    if order < 0:
        raise ValueError("order must be a positive integer")
    if order >= radii.shape[0] // 2:
        raise IndexError(
            f"radii does not have enough modes ({radii.shape[0]}) "
            f"to retain the lowest {order}."
        )
    spectrum = np.fft.fft(radii)
    if order == 0:
        spectrum[1:] = 0
    else:
        spectrum[order + 1 : -order] = 0
    return RadialBaselineFit(np.fft.ifft(spectrum).real, order)
