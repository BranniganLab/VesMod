"""Integration tests for experimental EdgeMod dynamic range selection."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vesmod.EdgeMod import Spectrum, SpectrumFitConfig
from vesmod.EdgeMod.experimental import QMinusThreeRangeSelector


def _spectrum_with_power_law() -> Spectrum:
    """Return a minimal Spectrum carrying q^-3 amplitudes from q=5 onward."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = 10.0
    spectrum.modes = np.arange(0, 12)
    spectrum.avg_amps2 = np.ones(12, dtype=float)
    q = spectrum.modes[3:].astype(float)
    spectrum.avg_amps2[3:] = q ** -3
    spectrum.avg_amps2[3] *= 8.0
    spectrum.avg_amps2[4] *= 0.2
    spectrum.kC = None
    spectrum.surface_tension = None
    spectrum.fit_result = None
    spectrum.fit_results = []
    return spectrum


def _selector() -> QMinusThreeRangeSelector:
    """Return standard experimental selection settings for these tests."""
    return QMinusThreeRangeSelector(
        lower_bound=3,
        upper_bound=12,
        min_modes=5,
        slope_tolerance=0.1,
        max_log_rmse=0.05,
    )


def test_selected_bounds_can_be_passed_to_core_physical_fit(monkeypatch):
    """Test experimental selection composes with the range-agnostic core fitter."""
    spectrum = _spectrum_with_power_law()
    selection = _selector().select(spectrum.modes, spectrum.avg_amps2)
    assert selection.accepted

    config = replace(
        SpectrumFitConfig(lmax=400, free_sigma=False),
        lower_bound=selection.lower_bound,
        upper_bound=selection.upper_bound,
    )
    calls = {}

    def fake_fit(fitting_range, lmax, free_sigma):
        calls["modes"] = fitting_range.modes.copy()
        calls["lmax"] = lmax
        calls["free_sigma"] = free_sigma
        return SimpleNamespace(best_values={"kC": 20.0, "sigma": 2.0})

    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.fit_spectrum_lmfit",
        fake_fit,
    )
    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.validate_lmfit_result",
        lambda result, fitting_range, free_sigma: None,
    )
    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.calc_tension_from_reduced_tension",
        lambda *args: 1.5,
    )

    fit = spectrum.extract_kc_from_fit(config)

    np.testing.assert_array_equal(calls["modes"], np.arange(5, 12))
    assert calls["lmax"] == 400
    assert calls["free_sigma"] is False
    assert fit.lower_bound == 5
    assert fit.upper_bound == 12
    assert fit.kC == 20.0
    assert fit.surface_tension == 1.5


def test_rejected_selection_does_not_require_core_physical_fit(monkeypatch):
    """Test experimental rejection can stop before the core fitter is invoked."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = 10.0
    spectrum.modes = np.arange(0, 12)
    spectrum.avg_amps2 = np.ones(12, dtype=float)
    q = spectrum.modes[3:].astype(float)
    spectrum.avg_amps2[3:] = q ** -2
    spectrum.kC = 99.0
    spectrum.surface_tension = 98.0
    spectrum.fit_result = None
    spectrum.fit_results = []

    selection = _selector().select(spectrum.modes, spectrum.avg_amps2)

    assert not selection.accepted
    assert "No trusted q range" in selection.reason
    assert spectrum.kC == 99.0
    assert spectrum.surface_tension == 98.0
    assert spectrum.fit_results == []


def test_core_spectrum_serialization_has_no_dynamic_selection_state():
    """Test experimental diagnostics do not become permanent Spectrum state."""
    spectrum = _spectrum_with_power_law()
    selection = _selector().select(spectrum.modes, spectrum.avg_amps2)
    assert selection.accepted

    data = spectrum._to_dict(include_arrays=False)

    assert "fit_range_selection" not in data
    assert "dynamic_range_selection" not in data
    assert selection.lower_bound == 5
    assert selection.upper_bound == 12
