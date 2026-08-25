"""Tests for preserving multiple core EdgeMod fit results on one Spectrum."""

from types import SimpleNamespace

import numpy as np

from vesmod.EdgeMod import Spectrum, SpectrumFit, SpectrumFitConfig


def _spectrum() -> Spectrum:
    """Return a minimal spectrum for repeated physical fitting."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = 10.0
    spectrum.modes = np.arange(0, 12)
    spectrum.avg_amps2 = np.ones(12, dtype=float)
    spectrum.kC = None
    spectrum.surface_tension = None
    spectrum.fit_result = None
    spectrum.fit_results = []
    return spectrum


def _mock_physical_fit(monkeypatch) -> None:
    """Replace physical fitting with a deterministic result keyed to first q."""
    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.fit_spectrum_lmfit",
        lambda fitting_range, lmax, free_sigma: SimpleNamespace(
            best_values={
                "kC": float(fitting_range.modes[0]),
                "sigma": 2.0,
            }
        ),
    )
    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.validate_lmfit_result",
        lambda result, fitting_range, free_sigma: None,
    )
    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.calc_tension_from_reduced_tension",
        lambda r0, reduced_sigma, kc, temperature: kc / 10.0,
    )


def test_multiple_physical_fits_are_both_preserved(monkeypatch):
    """Test a later physical fit does not replace an earlier fit result."""
    spectrum = _spectrum()
    first_config = SpectrumFitConfig(lower_bound=3, upper_bound=8)
    second_config = SpectrumFitConfig(lower_bound=5, upper_bound=12)
    _mock_physical_fit(monkeypatch)

    first = spectrum.extract_kc_from_fit(first_config)
    second = spectrum.extract_kc_from_fit(second_config)

    assert isinstance(first, SpectrumFit)
    assert isinstance(second, SpectrumFit)
    assert first.lower_bound == 3
    assert first.upper_bound == 8
    assert first.kC == 3.0
    assert second.lower_bound == 5
    assert second.upper_bound == 12
    assert second.kC == 5.0
    assert spectrum.fit_results == [first, second]
    assert spectrum.kC == second.kC
    assert spectrum.surface_tension == second.surface_tension


def test_spectrum_fit_supports_tuple_unpacking():
    """Test callers can unpack a SpectrumFit as kc and tension."""
    config = SpectrumFitConfig()
    fit = SpectrumFit(
        kC=20.0,
        surface_tension=1.5,
        lower_bound=3,
        upper_bound=8,
        config=config,
    )

    kc, tension = fit

    assert kc == 20.0
    assert tension == 1.5


def test_to_dict_serializes_all_physical_fit_results(monkeypatch):
    """Test repeated core analyses coexist in one serialized Spectrum."""
    spectrum = _spectrum()
    first_config = SpectrumFitConfig(lower_bound=3, upper_bound=8)
    second_config = SpectrumFitConfig(lower_bound=5, upper_bound=12)
    _mock_physical_fit(monkeypatch)

    spectrum.extract_kc_from_fit(first_config)
    spectrum.extract_kc_from_fit(second_config)

    data = spectrum._to_dict(include_arrays=False)

    assert len(data["fit_results"]) == 2
    assert data["fit_results"][0]["lower_bound"] == 3
    assert data["fit_results"][0]["upper_bound"] == 8
    assert data["fit_results"][0]["config"]["lmax"] == 500
    assert data["fit_results"][1]["lower_bound"] == 5
    assert data["fit_results"][1]["upper_bound"] == 12
    assert "method" not in data["fit_results"][0]
    assert "range_selection" not in data["fit_results"][1]
