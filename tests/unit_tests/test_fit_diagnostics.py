"""Fit estimates are retained without post-fit acceptance criteria."""

from types import SimpleNamespace

import numpy as np
import pytest
from lmfit import Parameters

from vesmod.EdgeMod import Spectrum, SpectrumEnsemble, SpectrumFitConfig
from vesmod.EdgeMod.spectrum_utils import (
    HSS97,
    MiniSpectrum,
    fit_spectrum_lmfit,
    fit_spectrum_to_theory_lmfit,
)


@pytest.mark.parametrize("entry_point", ["direct", "tuple", "spectrum", "ensemble"])
@pytest.mark.parametrize("condition", [
    "unsuccessful", "kc_bound", "sigma_bound", "missing_uncertainty",
    "large_uncertainty", "strong_correlation", "missing_correlation",
    "nonfinite_correlation",
])
def test_fit_estimates_are_retained(monkeypatch, caplog, entry_point, condition):
    """Each public fitting path returns results formerly rejected by validation."""
    params = Parameters()
    params.add("kC", value=25, min=1, max=500)
    params.add("sigma", value=2, min=-100, max=1000)
    params["kC"].stderr = 1.0
    params["kC"].correl = {"sigma": 0.5}
    if condition == "kc_bound":
        params["kC"].value = 500
    elif condition == "sigma_bound":
        params["sigma"].value = 1000
    elif condition == "missing_uncertainty":
        params["kC"].stderr = None
    elif condition == "large_uncertainty":
        params["kC"].stderr = 100.0
    elif condition == "strong_correlation":
        params["kC"].correl = {"sigma": -0.99}
    elif condition == "missing_correlation":
        params["kC"].correl = None
    elif condition == "nonfinite_correlation":
        params["kC"].correl = {"sigma": np.nan}
    result = SimpleNamespace(
        success=condition != "unsuccessful", message="evaluation limit reached",
        params=params, best_values=params.valuesdict(), best_fit=np.zeros(5),
        chisqr=1.0, redchi=1.0,
    )
    monkeypatch.setattr("vesmod.EdgeMod.spectrum_utils.Model.fit",
                        lambda *args, **kwargs: result)
    config = SpectrumFitConfig(lmax=50, free_sigma=True)
    group = MiniSpectrum(np.arange(3, 8), np.ones(5), None)

    with caplog.at_level("INFO", logger="vesmod.EdgeMod.spectrum_utils"):
        if entry_point == "direct":
            assert fit_spectrum_lmfit(group, 50, free_sigma=True, radii=[5.0]) is result
            kc = result.best_values["kC"]
        elif entry_point == "tuple":
            kc, sigma = fit_spectrum_to_theory_lmfit(group, 50, free_sigma=True, radii=[5.0])
            assert sigma == params["sigma"].value
        elif entry_point == "spectrum":
            spectrum = Spectrum.from_radii(np.full((2, 32), 5.0))
            spectrum.avg_amps2[:] = 1.0
            fit = spectrum.extract_kc_from_fit(config)
            assert spectrum.fit_results == [fit]
            assert spectrum.fit_result is result
            kc = fit.kC
        else:
            ensemble = SpectrumEnsemble()
            ensemble.add_spectrum(group.avg_amps2, group.modes, 5.0, r0=5.0)
            fit = ensemble.extract_kc_from_fit(config)
            assert ensemble.fit_results == [fit]
            assert fit.reduced_sigma == params["sigma"].value
            kc = fit.kC
    assert kc == params["kC"].value
    assert "Spectrum fit relative RMSE=1" in caplog.text


def test_real_lmfit_retains_strongly_correlated_estimates():
    """An exact HSS97 spectrum remains fit-able despite parameter tradeoff."""
    modes = np.arange(3, 8)
    group = MiniSpectrum(modes, HSS97(modes, 25, 100, 50), None)
    kc, sigma = fit_spectrum_to_theory_lmfit(group, 50, free_sigma=True, radii=[5.0])
    assert kc == pytest.approx(25, rel=1e-4)
    assert sigma == pytest.approx(100, rel=1e-4)


@pytest.mark.parametrize("prediction, expected", [(0.0, "nan"), (1.0, "inf")])
def test_zero_measured_power_logs_rmse(caplog, monkeypatch, prediction, expected):
    """Zero power has a diagnostic RMSE without an acceptance rule."""
    group = MiniSpectrum(np.arange(3, 8), np.zeros(5), None)
    result = SimpleNamespace(best_fit=np.full(5, prediction))
    monkeypatch.setattr("vesmod.EdgeMod.spectrum_utils.Model.fit",
                        lambda *args, **kwargs: result)
    with caplog.at_level("INFO", logger="vesmod.EdgeMod.spectrum_utils"):
        assert fit_spectrum_lmfit(group, 50, radii=[5.0]) is result
    assert f"Spectrum fit relative RMSE={expected}" in caplog.text


def test_optimizer_exceptions_still_propagate(monkeypatch):
    """Removing result validation does not turn optimizer exceptions into fits."""
    def fail(*args, **kwargs):
        raise ValueError("optimizer failed to evaluate model")

    monkeypatch.setattr("vesmod.EdgeMod.spectrum_utils.Model.fit", fail)
    group = MiniSpectrum(np.arange(3, 8), np.ones(5), None)
    with pytest.raises(ValueError, match="optimizer failed to evaluate model"):
        fit_spectrum_lmfit(group, 50, radii=[5.0])
