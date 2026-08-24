"""Tests for preserving multiple EdgeMod fit results on one Spectrum."""

import numpy as np

from vesmod.EdgeMod import (
    FixedFitRangeSelector,
    QMinusThreeFitRangeSelector,
    Spectrum,
    SpectrumFit,
    SpectrumFitConfig,
)


def _spectrum() -> Spectrum:
    """Return a minimal spectrum with q^-3 scaling from q=5 onward."""
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


def test_fixed_and_dynamic_fits_are_both_preserved(monkeypatch):
    """Test a later fit does not replace an earlier fit result."""
    spectrum = _spectrum()
    fixed_config = SpectrumFitConfig(
        range_selector=FixedFitRangeSelector(3, 8),
    )
    dynamic_config = SpectrumFitConfig(
        range_selector=QMinusThreeFitRangeSelector(
            lower_bound=3,
            upper_bound=12,
            min_modes=5,
            slope_tolerance=0.1,
            max_log_rmse=0.05,
        )
    )

    def fake_fit(fitting_range, lmax, free_sigma):
        return float(fitting_range.modes[0]), 2.0

    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.fit_spectrum_to_theory_lmfit",
        fake_fit,
    )
    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.calc_tension_from_reduced_tension",
        lambda r0, reduced_sigma, kc, temperature: kc / 10.0,
    )

    fixed = spectrum.extract_kc_from_fit(fixed_config)
    dynamic = spectrum.extract_kc_from_fit(dynamic_config)

    assert isinstance(fixed, SpectrumFit)
    assert isinstance(dynamic, SpectrumFit)
    assert fixed.method == "FixedFitRangeSelector"
    assert fixed.lower_bound == 3
    assert fixed.upper_bound == 8
    assert fixed.kC == 3.0
    assert dynamic.method == "QMinusThreeFitRangeSelector"
    assert dynamic.lower_bound == 5
    assert dynamic.upper_bound == 12
    assert dynamic.kC == 5.0
    assert spectrum.fit_results == [fixed, dynamic]
    assert spectrum.kC == dynamic.kC
    assert spectrum.surface_tension == dynamic.surface_tension


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


def test_to_dict_serializes_all_fit_results(monkeypatch):
    """Test fixed and dynamic analyses coexist in one serialized Spectrum."""
    spectrum = _spectrum()
    fixed_config = SpectrumFitConfig(
        range_selector=FixedFitRangeSelector(3, 8),
    )
    dynamic_config = SpectrumFitConfig(
        range_selector=QMinusThreeFitRangeSelector(
            lower_bound=3,
            upper_bound=12,
            min_modes=5,
            slope_tolerance=0.1,
            max_log_rmse=0.05,
        )
    )

    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.fit_spectrum_to_theory_lmfit",
        lambda fitting_range, lmax, free_sigma: (
            float(fitting_range.modes[0]),
            2.0,
        ),
    )
    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.calc_tension_from_reduced_tension",
        lambda r0, reduced_sigma, kc, temperature: kc / 10.0,
    )

    spectrum.extract_kc_from_fit(fixed_config)
    spectrum.extract_kc_from_fit(dynamic_config)

    data = spectrum._to_dict(include_arrays=False)

    assert len(data["fit_results"]) == 2
    assert data["fit_results"][0]["method"] == "FixedFitRangeSelector"
    assert data["fit_results"][0]["lower_bound"] == 3
    assert data["fit_results"][0]["config"]["lmax"] == 500
    assert data["fit_results"][1]["method"] == "QMinusThreeFitRangeSelector"
    assert data["fit_results"][1]["lower_bound"] == 5
    assert data["fit_results"][1]["range_selection"]["accepted"] is True
    assert (
        data["fit_results"][1]["config"]["range_selector"]["min_modes"]
        == 5
    )
