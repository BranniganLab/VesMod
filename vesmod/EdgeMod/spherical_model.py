"""Spherical model for vesicle contour fluctuation spectra."""

from functools import lru_cache
from numbers import Integral
import numpy as np
from scipy.special import gammaln

from .camera_integration import exposure_power_factor
from .config import SpectrumFitConfig
from .relaxation import _relaxation_times_for_validated_modes, mode_relaxation_times
from .spherical_validation import validate_spherical_inputs, validate_vesicle_radii


def spherical_contour_spectrum(q: list[int], kC: float, sigma: float, lmax: int) -> list[float]:
    """Predict mean contour power from instantaneous snapshots of a spherical vesicle.

    The vesicle fluctuates at thermal equilibrium. The model predicts the
    mean squared amplitude of each contour mode across snapshots, without
    averaging motion within a camera exposure. See Hackl, Seifert, and
    Sackmann (1997), Eqs. 7 and 8.

    Parameters
    ----------
    q : list[int]
        Wave number / independent variable.
    kC : float
        Bending modulus to be fit.
    sigma : float
        Effective tension to be fit.
    lmax : int
        Inclusive upper value of the summation.

    Returns
    -------
    function : list
        The values to be fit.

    Notes
    -----
    Public calls require physical Fourier modes q >= 2 and q <= lmax.

    """
    modes = validate_spherical_inputs(q, kC, sigma, lmax)
    function = []
    lmax = int(lmax)
    for wavenum in modes:
        summ = 0.0
        for l in range(wavenum, lmax + 1):
            denom = (l - 1) * (l + 2) * (l ** 2 + l + sigma)
            if denom == 0:
                raise ValueError(
                    "spherical_contour_spectrum denominator is zero for the requested sigma and mode."
                )
            summ += Nlq_Plq0_squared(l, wavenum) / denom
        function.append((1 / kC) * summ)
    return function


@lru_cache(maxsize=256)
def _camera_projection(q, lmax):
    """Cache fixed Nlq Plq^2 geometry for repeated optimizer evaluations.

    Parameters
    ----------
    q : int
        Contour Fourier mode.
    lmax : int
        Inclusive maximum spherical harmonic degree.

    Returns
    -------
    tuple
        Contributing degrees and their contour projection weights.
    """
    ell = np.arange(q, lmax + 1, 2, dtype=float)
    weights = np.array([Nlq_Plq0_squared(int(l), q) for l in ell])
    return ell, weights


def predict_spherical_contour_power(
    q, kC, sigma, lmax, *, config, radii=None, relaxation=mode_relaxation_times,
    camera_integration=exposure_power_factor,
):
    """Predict dimensionless complex contour power for fitting or plotting.

    With camera integration disabled, return the instantaneous spherical
    spectrum from ``spherical_contour_spectrum``. With it enabled, calculate how
    much power survives exposure averaging using the selected relaxation law.
    The camera integration callable converts exposure-to-relaxation ratios to
    retained power fractions before projection onto the contour. Predictions
    average equally across replica radii with shared kC and reduced tension.

    The relaxation callable returns times in seconds with shape (radii, ell)
    for the supplied spherical modes, trial parameters, config, and radii.

    Parameters
    ----------
    q : array-like
        Contour Fourier modes.
    kC : float
        Bending modulus in kBT.
    sigma : float
        Dimensionless reduced tension.
    lmax : int
        Inclusive spherical harmonic summation limit.
    config : SpectrumFitConfig
        Fit settings including exposure and relaxation choices.
    radii : float, array-like, or None
        Replica radii in micrometers, required for camera integration.
    relaxation : callable
        Callable returning an array of times in seconds indexed by radius and degree.
    camera_integration : callable
        Callable mapping exposure-to-relaxation ratios to retained fractions.

    Returns
    -------
    list of float
        Dimensionless contour powers in the order of q.
    """
    modes = validate_spherical_inputs(q, kC, sigma, lmax)
    if not config.camera_integration:
        return spherical_contour_spectrum(modes, kC, sigma, lmax)
    projections = [_camera_projection(mode, int(lmax)) for mode in modes]
    spherical_modes = np.unique(np.concatenate([ell for ell, _ in projections])).astype(int)
    # Projection produces valid degrees from the checked contour modes and lmax.
    calculate_times = (
        _relaxation_times_for_validated_modes if relaxation is mode_relaxation_times else relaxation
    )
    relaxation_times = calculate_times(spherical_modes, kC, sigma, config=config, radii=radii)
    predictions = []
    for ell, weights in projections:
        stiffness = (ell - 1) * (ell + 2) * (ell * (ell + 1) + sigma)
        times = relaxation_times[:, np.searchsorted(spherical_modes, ell)]
        retained = camera_integration(config.exposure_time / times).mean(axis=0)
        predictions.append(float(np.sum(weights * retained / stiffness) / kC))
    return predictions


def Nlq_Plq0_squared(l: int, q: int) -> float:
    """Return the squared normalized associated Legendre value at zero.

    The direct product of the spherical-harmonic normalization and
    scipy.special.lpmv is numerically unstable at high l and q: the
    normalization can underflow while the polynomial overflows. This
    implementation combines their analytic factorial expressions in log space
    so only the finite final value is exponentiated.

    Parameters
    ----------
    l : int
        Polynomial order.
    q : int
        Polynomial degree / wave number.

    Returns
    -------
    float
        The squared normalized associated Legendre value for contour projection.

    """
    for name, value in (("l", l), ("q", q)):
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
    if q < 0:
        raise ValueError("q must be non-negative.")
    if l < 0:
        raise ValueError("l must be non-negative.")
    if q > l:
        raise ValueError("q must be <= l.")

    l = int(l)
    q = int(q)
    if (l + q) % 2:
        return 0.0

    half_difference = (l - q) // 2
    half_sum = (l + q) // 2
    log_total = (
        np.log(2 * l + 1)
        - np.log(4 * np.pi)
        + gammaln(l - q + 1)
        + gammaln(l + q + 1)
        - 2 * l * np.log(2)
        - 2 * gammaln(half_difference + 1)
        - 2 * gammaln(half_sum + 1)
    )
    return float(np.exp(log_total))


def configure_spectrum_model(
    config: SpectrumFitConfig, radii, *, expected_replica_count=None, relaxation=mode_relaxation_times,
    camera_integration=exposure_power_factor,
):
    """Return a predictor and the bending-modulus and reduced-tension bounds.

    The fitter calls the returned predictor repeatedly with the signature
    ``(q, kC, sigma, lmax)``. The nested function binds
    the configuration, validated replica radii, relaxation law, and camera
    integration function so those inputs stay fixed during optimization.

    Parameters
    ----------
    config : SpectrumFitConfig
        Fit settings. Set camera_integration=False for the instantaneous spectrum.
    radii : float or array-like
        Replica radii in micrometers.
    expected_replica_count : int or None
        Optional number of expected radii.
    relaxation : callable
        Callable calculating spherical-mode relaxation times.
    camera_integration : callable
        Callable mapping exposure-to-relaxation ratios to retained fractions.

    Returns
    -------
    tuple
        Predictor, bending modulus bounds in kBT, and dimensionless reduced tension bounds.
    """
    if not isinstance(config, SpectrumFitConfig):
        raise TypeError("config must be a SpectrumFitConfig.")
    radii = validate_vesicle_radii(radii)
    if expected_replica_count is not None:
        if (isinstance(expected_replica_count, (bool, np.bool_))
                or not isinstance(expected_replica_count, (int, np.integer))
                or expected_replica_count < 1):
            raise ValueError("expected_replica_count must be a positive integer.")
        if radii.size != expected_replica_count:
            raise ValueError(
                "Spectrum fitting requires one radius per replica: "
                f"expected {expected_replica_count} radii, received {radii.size}. "
                "Supply each replica's radius, rather than a mean radius."
            )

    def predict(q, kC, sigma, lmax):
        """Evaluate spherical contour powers with the captured fit settings.

        Parameters
        ----------
        q : array-like
            Contour Fourier modes.
        kC : float
            Bending modulus in kBT.
        sigma : float
            Dimensionless reduced tension.
        lmax : int
            Inclusive spherical harmonic summation limit.

        Returns
        -------
        list of float
            Predicted contour powers for the bound settings and radii.
        """
        return predict_spherical_contour_power(
            q, kC, sigma, lmax, config=config, radii=radii,
            relaxation=relaxation, camera_integration=camera_integration,
        )

    # Exposure averaging requires stability of the spherical l=2 mode.
    sigma_min = -6 + 1e-8 if config.camera_integration else -100
    sigma_max = 100 * float(np.mean(radii))**2
    return predict, (1, 500), (sigma_min, sigma_max)
