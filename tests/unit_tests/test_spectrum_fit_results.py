"""Tests for preserving multiple core EdgeMod fit results on one Spectrum."""

from spectrum_fixture import SyntheticSpectrum

from types import SimpleNamespace

import numpy as np

from vesmod.edgemod import Spectrum, SpectrumFit, SpectrumFitConfig


def _spectrum() -> Spectrum:
    """Return a minimal spectrum for repeated physical fitting."""
    spectrum = SyntheticSpectrum.__new__(SyntheticSpectrum)
    spectrum.r0 = 10.0
    spectrum.modes = np.arange(0, 12)
    spectrum.powers = np.ones(12, dtype=float)
    spectrum.kC = None
    spectrum.surface_tension = None
    spectrum.fit_result = None
    spectrum.fit_results = []
    return spectrum


def _mock_physical_fit(monkeypatch) -> None:
    """Replace physical fitting with a deterministic result keyed to first q."""
    monkeypatch.setattr(
        "vesmod.edgemod.fitting.optimization.least_squares.fit_spectrum",
        lambda modes, powers, lmax, free_sigma, **kwargs: SimpleNamespace(
            best_values={
                "kC": float(modes[0]),
                "sigma": 2.0,
            }
        ),
    )
    monkeypatch.setattr(
        "vesmod.edgemod.spectra.spectrum.calc_tension_from_reduced_tension",
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

    data = spectrum.to_dict(include_arrays=False)

    assert len(data["fit_results"]) == 2
    assert data["fit_results"][0]["lower_bound"] == 3
    assert data["fit_results"][0]["upper_bound"] == 8
    assert data["fit_results"][0]["config"]["lmax"] == 500
    assert data["fit_results"][1]["lower_bound"] == 5
    assert data["fit_results"][1]["upper_bound"] == 12
    assert "method" not in data["fit_results"][0]
    assert "range_selection" not in data["fit_results"][1]


def test_fit_records_temporal_variance_definition(monkeypatch):
    """The exported fit identifies which power was supplied to the fitter."""
    spectrum = _spectrum()
    spectrum.power_definition = "temporal_variance"
    _mock_physical_fit(monkeypatch)

    fit = spectrum.extract_kc_from_fit()

    assert fit.power_definition == "temporal_variance"
    assert fit.to_dict()["power_definition"] == "temporal_variance"


def test_weighted_fit_diagnostics_round_trip(tmp_path):
    """An actual weighted optimizer fit retains ordered covariance and settings."""
    import json

    from vesmod.edgemod import EnsembleMember
    from vesmod.edgemod.fitting.fitting_functions.spherical_model import spherical_contour_spectrum

    spectrum = Spectrum.from_radii(np.full((4, 32), 5.0))
    selected = (spectrum.modes >= 3) & (spectrum.modes < 8)
    spectrum.powers[selected] = (
        np.asarray(spherical_contour_spectrum(spectrum.modes[selected], 25, 0, 30))
        * np.array([0.98, 1.01, 1.00, 1.02, 0.99])
    )
    errors = np.linspace(1e-5, 2e-5, len(spectrum.modes))
    config = SpectrumFitConfig(lmax=30, free_sigma=False, kC_start=20, max_nfev=150)
    fit = spectrum.extract_kc_from_fit(config, power_errors=errors.tolist())

    assert fit.reduced_sigma == 0
    assert fit.diagnostics.success is True
    assert fit.diagnostics.parameter_names == ("kC",)
    assert fit.diagnostics.parameter_stderr[0] > 0
    np.testing.assert_allclose(fit.diagnostics.covariance, spectrum.fit_result.covar)
    np.testing.assert_allclose(fit.diagnostics.weights, 1 / errors[selected])
    assert fit.diagnostics.rmse > 0
    assert fit.config.max_nfev == 150

    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps(fit.to_dict()), encoding="utf-8")
    assert SpectrumFit.from_json(fit_path) == fit
    spectrum_path = tmp_path / "spectrum.json"
    spectrum.to_json(spectrum_path)
    assert EnsembleMember.from_json(spectrum_path).fit_results == (fit,)


def test_unversioned_spectrum_fit_restores_unavailable_diagnostics():
    """Historical fit exports load without fabricating optimizer results."""
    saved = {
        "kC": 22.0,
        "surface_tension": 0.0,
        "lower_bound": 3,
        "upper_bound": 8,
        "config": {},
    }
    restored = SpectrumFit.from_dict(saved)
    assert restored.kC == 22
    assert restored.diagnostics is None
    assert restored.reduced_sigma is None
    assert restored.vesmod_version is None
