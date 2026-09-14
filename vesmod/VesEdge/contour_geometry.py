"""Shared geometry helpers for uniformly sampled radial contours.

The functions in this module operate on contours represented as radii sampled
at evenly spaced angles from zero through (but not including) 2π. They are
deliberately independent of extraction, QC, and persistence so the same
geometric projection can be reused by each layer.
"""

from dataclasses import dataclass

import cv2
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


def radial_contour_centroid(origin, radii):
    """Return the area centroid of a uniformly sampled radial contour.

    ``radii`` is interpreted on a uniform angular grid about ``origin``. The
    corresponding Cartesian polygon is passed to OpenCV for its spatial
    moments. Moments are evaluated in coordinates relative to ``origin`` so a
    rigid translation of the same contour does not lose precision when OpenCV
    converts contour coordinates to single precision.

    Parameters
    ----------
    origin : tuple[float, float]
        Cartesian ``(x, y)`` origin used by the radial representation.
    radii : numpy.ndarray
        Positive radial distances sampled uniformly over ``[0, 2π)``.

    Returns
    -------
    tuple[float, float]
        Area centroid of the contour in Cartesian ``(x, y)`` coordinates.

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
    x_relative = radii * np.cos(theta)
    y_relative = radii * np.sin(theta)

    contour = np.column_stack((x_relative, y_relative)).astype(np.float32)
    moments = cv2.moments(contour)
    if np.isclose(moments["m00"], 0.0):
        raise ValueError("contour must enclose non-zero area")

    centroid_relative_x = float(moments["m10"] / moments["m00"])
    centroid_relative_y = float(moments["m01"] / moments["m00"])
    return (
        origin[0] + centroid_relative_x,
        origin[1] + centroid_relative_y,
    )
