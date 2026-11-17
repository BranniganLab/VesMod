"""Power attenuation from finite camera exposure."""

import numpy as np


def exposure_power_factor(x):
    """Return retained power for a boxcar-averaged exponential correlation.

    For ``x = exposure_time / relaxation_time > 0``, the retained power
    fraction is ``2 * (x - 1 + exp(-x)) / x**2``. Its limit at ``x = 0``
    is 1.

    For ``x < 1e-3``, the Taylor approximation
    ``1 - x/3 + x**2/12 - x**3/60 + x**4/360`` avoids cancellation.
    For larger ratios, the equivalent expression
    ``(2/x) * (1 + expm1(-x)/x)`` avoids squaring large x.

    Parameters
    ----------
    x : float or array-like
        Scalar or array of exposure time divided by relaxation time.

    Returns
    -------
    numpy.ndarray
        Retained power fraction with the same shape as x.
    """
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("Exposure/relaxation ratio must be finite and nonnegative.")
    result = np.empty_like(x)
    small = x < 1e-3
    z = x[small]
    result[small] = 1 - z / 3 + z**2 / 12 - z**3 / 60 + z**4 / 360
    z = x[~small]
    result[~small] = 2 / z * (1 + np.expm1(-z) / z)
    return result
