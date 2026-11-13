"""Power attenuation from finite camera exposure."""

import numpy as np


def exposure_power_factor(x):
    """Return retained power for a boxcar-averaged exponential correlation.

    ``x`` is exposure time / relaxation time. A series avoids cancellation
    near zero; the reciprocal form avoids squaring large x.
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
