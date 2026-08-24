"""Integration tests for dynamic EdgeMod spectrum fit-range selection."""

import numpy as np
import pytest

from vesmod.EdgeMod import (
    QMinusThreeFitRangeSelector,
    Spectrum,
    SpectrumFitConfig,
)


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
    spectrum.fit_range_selection = None
    spectrum.fit_results = []
    return spectrum


def _dynamic_config() -> SpectrumFitConfig:
    """Return standard dynamic-selection settings for these tests."""
    return SpectrumFitConfig(
        lmax=400,
        free_sigma=False,
        range_selector=QMinusThreeFitRangeSelector(
            lower_bound=3,
            upper_bound=12,
            min_modes=5,
            slope_tolerance=0.1,
            max_log_rmse=0.05,
        ),
    )


def test_extract_kc_from_fit_uses_dynamic_selected_range(monkeypatch):
    """Test the physical fitter receives only the dynamically selected modes."""
    spectrum = _spectrum_with_power_law()
    config = _dynamic_config()
    calls = {}

    def fake_fit(fitting_range, lmax, free_sigma):
        calls["modes"] = fitting_range.modes.copy()
        calls["lmax"] = lmax
        calls["free_sigma"] = free_sigma
        return 20.0, 2.0

    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.fit_spectrum_to_theory_lmfit",
        fake_fit,
    )
    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.calc_tension_from_reduced_tension",
        lambda *args: 1.5,
    )

    fit = spectrum.extract_kc_from_fit(config)

    np.testing.assert_array_equal(calls["modes"], np.arange(5, 12))
    assert calls["lmax"] == 400
    assert calls["free_sigma"] is False
    assert spectrum.fit_range_selection is not None
    assert spectrum.fit_range_selection.accepted
    assert spectrum.fit_range_selection.lower_bound == 5
    assert spectrum.fit_range_selection.upper_bound == 12
    assert fit.kC == 20.0
    assert fit.surface_tension == 1.5
    assert fit.config is config


def test_extract_kc_from_fit_rejects_before_physical_fit(monkeypatch):
    """Test rejection preserves the most recent successful fit values."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = 10.0
    spectrum.modes = np.arange(0, 12)
    spectrum.avg_amps2 = np.ones(12, dtype=float)
    q = spectrum.modes[3:].astype(float)
    spectrum.avg_amps2[3:] = q ** -2
    spectrum.kC = 99.0
    spectrum.surface_tension = 98.0
    spectrum.fit_range_selection = None
    spectrum.fit_results = []
    config = _dynamic_config()

    def fail_if_called(*args, **kwargs):
        pytest.fail("Physical spectrum fitter should not run after range rejection.")

    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.fit_spectrum_to_theory_lmfit",
        fail_if_called,
    )

    with pytest.raises(ValueError, match="No trusted q range"):
        spectrum.extract_kc_from_fit(config)

    assert spectrum.fit_range_selection is not None
    assert not spectrum.fit_range_selection.accepted
    assert spectrum.kC == 99.0
    assert spectrum.surface_tension == 98.0
    assert spectrum.fit_results == []


def test_to_dict_serializes_dynamic_range_diagnostics():
    """Test dynamic selection diagnostics are available to later reporting."""
    spectrum = _spectrum_with_power_law()
    selector = _dynamic_config().range_selector
    spectrum.fit_range_selection = selector.select(
        spectrum.modes,
        spectrum.avg_amps2,
    )

    data = spectrum._to_dict(include_arrays=False)

    assert data["fit_range_selection"]["accepted"] is True
    assert data["fit_range_selection"]["lower_bound"] == 5
    assert data["fit_range_selection"]["upper_bound"] == 12
    assert data["fit_range_selection"]["slope"] == pytest.approx(-3.0)
    assert data["fit_range_selection"]["log_rmse"] == pytest.approx(
        0.0,
        abs=1e-12,
    )
