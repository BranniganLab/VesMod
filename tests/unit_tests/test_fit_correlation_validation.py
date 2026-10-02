"""Validate the kC–reduced-sigma identifiability threshold."""
from types import SimpleNamespace

import numpy as np
import pytest
from lmfit import Parameters

from vesmod.EdgeMod.spectrum_utils import (
    MiniSpectrum, HSS97, fit_spectrum_lmfit, validate_lmfit_result,
)


def fit_with_correlation(correlation):
    """All other checks pass, isolating the correlation criterion."""
    params = Parameters()
    params.add('kC', value=25, min=1, max=500)
    params.add('sigma', value=2, min=-100, max=1000)
    params['kC'].stderr = 1.0
    params['kC'].correl = None if correlation is None else {'sigma': correlation}
    group = MiniSpectrum(np.arange(3, 8), np.ones(5), None)
    result = SimpleNamespace(success=True, params=params, best_fit=np.ones(5))
    return result, group


@pytest.mark.parametrize('correlation', [-.951, .951, -1.0, 1.0])
def test_rejects_strong_correlation(correlation):
    result, group = fit_with_correlation(correlation)
    with pytest.raises(ValueError, match='strongly correlated kC and sigma'):
        validate_lmfit_result(result, group, free_sigma=True)


@pytest.mark.parametrize('correlation', [-.95, .95, -.5, .5, 0.0])
def test_accepts_correlation_at_or_below_threshold(correlation):
    result, group = fit_with_correlation(correlation)
    validate_lmfit_result(result, group, free_sigma=True)


@pytest.mark.parametrize('correlation', [None, np.nan, np.inf, -np.inf])
def test_requires_finite_correlation_when_sigma_is_free(correlation):
    result, group = fit_with_correlation(correlation)
    with pytest.raises(ValueError, match='finite kC-sigma correlation'):
        validate_lmfit_result(result, group, free_sigma=True)


def test_fixed_sigma_does_not_require_correlation():
    result, group = fit_with_correlation(None)
    validate_lmfit_result(result, group, free_sigma=False)


def test_real_lmfit_covariance_rejects_degenerate_spectrum():
    """A converged exact fit can still have a strong parameter tradeoff."""
    q = np.arange(3, 8)
    group = MiniSpectrum(q, HSS97(q, 25, 100, 50), None)
    result = fit_spectrum_lmfit(group, 50, free_sigma=True)
    assert abs(result.params['kC'].correl['sigma']) > .95
    with pytest.raises(ValueError, match='strongly correlated kC and sigma'):
        validate_lmfit_result(result, group, free_sigma=True)


@pytest.mark.parametrize('free_sigma', [False, True])
def test_large_residuals_do_not_reject_fit(free_sigma):
    """Residual magnitude is not an acceptance criterion."""
    result, group = fit_with_correlation(.5)
    result.best_fit = np.zeros(5)  # Relative RMSE is 1, above the former 0.25 limit.
    validate_lmfit_result(result, group, free_sigma=free_sigma)
