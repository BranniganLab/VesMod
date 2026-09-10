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


def recenter_radial_contour(origin, radii):
    """Re-express a radial contour about the centroid of its enclosed area.

    ``radii`` is interpreted on a uniform angular grid about ``origin``. The
    corresponding Cartesian polygon is used to calculate the standard area
    centroid. The same detected boundary is then converted back to polar
    coordinates about that centroid and interpolated onto a uniform angular
    grid with the original number of samples.

    Parameters
    ----------
    origin : tuple[float, float]
        Cartesian ``(x, y)`` origin used by the input radial representation.
    radii : numpy.ndarray
        Positive radial distances sampled uniformly over ``[0, 2π)``.

    Returns
    -------
    tuple[tuple[float, float], numpy.ndarray]
        The contour-area centroid in Cartesian ``(x, y)`` coordinates and the
        radial values re-expressed about that centroid.

    Raises
    ------
    TypeError
        If ``radii`` is not a one-dimensional NumPy array.
    ValueError
        If fewer than three samples are provided or the contour encloses zero
        area.
    """
    if not isinstance(radii, np.ndarray) or radii.ndim != 1:
        raise TypeError("radii must be a 1D numpy array")
    if radii.size < 3:
        raise ValueError("radii must contain at least three samples")

    radii = np.asarray(radii, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, radii.size, endpoint=False)
    x = origin[0] + radii * np.cos(theta)
    y = origin[1] + radii * np.sin(theta)

    next_x = np.roll(x, -1)
    next_y = np.roll(y, -1)
    cross = x * next_y - next_x * y
    signed_area_twice = float(np.sum(cross))
    if np.isclose(signed_area_twice, 0.0):
        raise ValueError("contour must enclose non-zero area")

    centroid_x = float(
        np.sum((x + next_x) * cross) / (3.0 * signed_area_twice)
    )
    centroid_y = float(
        np.sum((y + next_y) * cross) / (3.0 * signed_area_twice)
    )
    centroid = (centroid_x, centroid_y)

    dx = x - centroid_x
    dy = y - centroid_y
    new_theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    new_radii = np.hypot(dx, dy)

    order = np.argsort(new_theta)
    target_theta = np.linspace(0.0, 2.0 * np.pi, radii.size, endpoint=False)
    uniform_radii = np.interp(
        target_theta,
        new_theta[order],
        new_radii[order],
        period=2.0 * np.pi,
    )

    return centroid, uniform_radii
