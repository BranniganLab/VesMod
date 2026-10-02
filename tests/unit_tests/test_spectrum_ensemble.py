#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for SpectrumEnsemble."""

import numpy as np
import pytest

from vesmod.EdgeMod import EnsembleFit, SpectrumFitConfig
from vesmod.EdgeMod.spectrum_ensemble import SpectrumEnsemble


def test_new_spectrum_ensemble_is_empty():
    """Test that a new SpectrumEnsemble starts with no spectra, no kC values, and no modes."""
    avg = SpectrumEnsemble()

    assert len(avg) == 0
    assert avg.spectra_list == []
    assert avg.kC_list == []
    assert avg.modes is None
    assert avg.fit_results == []


def test_public_edge_mod_import_exposes_spectrum_ensemble():
    """Test that SpectrumEnsemble is part of the public EdgeMod API."""
    from vesmod.EdgeMod import SpectrumEnsemble as PublicSpectrumEnsemble

    assert PublicSpectrumEnsemble is SpectrumEnsemble


def test_add_spectrum_stores_first_spectrum_and_modes():
    """Test that the first added spectrum defines the stored mode array and is saved."""
    avg = SpectrumEnsemble()

    avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 2, 3], kC=20.0)

    assert len(avg) == 1
    np.testing.assert_array_equal(avg.modes, np.array([1, 2, 3]))
    assert avg.spectra_list == [[1.0, 2.0, 3.0]]
    assert avg.kC_list == [20.0]


def test_add_spectrum_accepts_matching_modes_after_first_spectrum():
    """Test that additional spectra are accepted when their modes match the stored modes exactly."""
    avg = SpectrumEnsemble()

    avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 2, 3], kC=20.0)
    avg.add_spectrum(avg_amps2=[2.0, 4.0, 6.0], modes=[1, 2, 3], kC=24.0)

    assert len(avg) == 2
    np.testing.assert_array_equal(avg.modes, np.array([1, 2, 3]))
    assert avg.spectra_list == [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]
    assert avg.kC_list == [20.0, 24.0]


def test_add_spectrum_rejects_nonmatching_modes():
    """Test that add_spectrum raises ValueError when a later spectrum uses different modes."""
    avg = SpectrumEnsemble()

    avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 2, 3], kC=20.0)

    with pytest.raises(ValueError, match=r"\[1, 3, 5\] does not equal \[1 2 3\]"):
        avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 3, 5], kC=22.0)


def test_add_spectrum_rejects_invalid_internal_modes_state():
    """Test that add_spectrum raises TypeError if self.modes has an invalid internal type."""
    avg = SpectrumEnsemble()
    avg.modes = [1, 2, 3]

    with pytest.raises(TypeError, match="self.modes must be ndarray or None"):
        avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 2, 3], kC=20.0)


def test_avg_amps2_returns_elementwise_mean_across_spectra():
    """Test that avg_amps2 returns the elementwise mean of all stored spectra."""
    avg = SpectrumEnsemble()

    avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 2, 3], kC=20.0)
    avg.add_spectrum(avg_amps2=[3.0, 4.0, 5.0], modes=[1, 2, 3], kC=22.0)
    avg.add_spectrum(avg_amps2=[5.0, 6.0, 7.0], modes=[1, 2, 3], kC=24.0)

    np.testing.assert_allclose(avg.avg_amps2, np.array([3.0, 4.0, 5.0]))


def test_avg_amps2_std_returns_sample_standard_deviation_across_spectra():
    """Test that avg_amps2_std uses ddof=1 to calculate sample standard deviation."""
    avg = SpectrumEnsemble()

    spectra = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
        ]
    )
    for spectrum in spectra:
        avg.add_spectrum(avg_amps2=spectrum.tolist(), modes=[1, 2, 3], kC=20.0)

    expected = np.std(spectra, axis=0, ddof=1)
    np.testing.assert_allclose(avg.avg_amps2_std, expected)


def test_avg_amps2_ste_returns_standard_error_of_average_spectrum():
    """
    Test that avg_amps2_ste divides avg_amps2_std by the square root
    of the number of stored spectra.
    """
    avg = SpectrumEnsemble()

    spectra = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
        ]
    )

    for spectrum in spectra:
        avg.add_spectrum(
            avg_amps2=spectrum.tolist(),
            modes=[1, 2, 3],
            kC=20.0,
        )

    expected = np.std(spectra, axis=0, ddof=1) / np.sqrt(len(spectra))

    np.testing.assert_allclose(avg.avg_amps2_ste, expected)


def test_kC_std_returns_sample_standard_deviation_of_replica_kC_values():
    """Test that kC_std uses ddof=1 to calculate sample standard deviation of replica kC values."""
    avg = SpectrumEnsemble()

    avg.add_spectrum(avg_amps2=[1.0, 2.0], modes=[1, 2], kC=20.0)
    avg.add_spectrum(avg_amps2=[2.0, 3.0], modes=[1, 2], kC=22.0)
    avg.add_spectrum(avg_amps2=[3.0, 4.0], modes=[1, 2], kC=24.0)

    expected = np.std([20.0, 22.0, 24.0], ddof=1)
    assert avg.kC_std == pytest.approx(expected)


def test_kC_ste_returns_standard_error_of_replica_kC_values():
    """Test that kC_ste divides kC_std by sqrt of the number of replica kC values."""
    avg = SpectrumEnsemble()

    avg.add_spectrum(avg_amps2=[1.0, 2.0], modes=[1, 2], kC=20.0)
    avg.add_spectrum(avg_amps2=[2.0, 3.0], modes=[1, 2], kC=22.0)
    avg.add_spectrum(avg_amps2=[3.0, 4.0], modes=[1, 2], kC=24.0)

    expected = np.std([20.0, 22.0, 24.0], ddof=1) / np.sqrt(3)
    assert avg.kC_ste == pytest.approx(expected)


def test_isolate_mode_range_returns_selected_modes_and_average_amplitudes():
    """Test that _isolate_mode_range keeps modes >= lower_bound and < upper_bound."""
    avg = SpectrumEnsemble()

    avg.add_spectrum(
        avg_amps2=[10.0, 20.0, 30.0, 40.0], modes=[1, 2, 3, 4], kC=20.0
    )
    avg.add_spectrum(
        avg_amps2=[20.0, 40.0, 60.0, 80.0], modes=[1, 2, 3, 4], kC=22.0
    )

    mini_spectrum = avg._isolate_mode_range(lower_bound=2, upper_bound=4)

    np.testing.assert_array_equal(mini_spectrum.modes, np.array([2, 3]))
    np.testing.assert_allclose(mini_spectrum.avg_amps2, np.array([30.0, 45.0]))
    np.testing.assert_allclose(mini_spectrum.avg_amps2_ste, np.array([10.0, 15.0]))


def test_isolate_mode_range_raises_error_when_no_modes_have_been_added():
    """Test that _isolate_mode_range raises AttributeError before any spectrum has set modes."""
    avg = SpectrumEnsemble()

    with pytest.raises(AttributeError, match="There are no modes"):
        avg._isolate_mode_range(lower_bound=2, upper_bound=4)


def test_extract_kC_from_fit_uses_isolated_mode_range(monkeypatch):
    """Legacy fixed-sigma fitting receives the selected ensemble slice."""
    from types import SimpleNamespace

    avg = SpectrumEnsemble()
    avg.add_spectrum([10.0, 20.0, 30.0, 40.0], [1, 2, 3, 4], 20.0)
    avg.add_spectrum([20.0, 40.0, 60.0, 80.0], [1, 2, 3, 4], 22.0)
    calls = {}

    def fake_fit(group, lmax, free_sigma, weighted=False):
        calls.update(group=group, lmax=lmax, free_sigma=free_sigma, weighted=weighted)
        return SimpleNamespace(best_values={"kC": 123.0, "sigma": 0.0}, chisqr=1.0, redchi=1.0)

    monkeypatch.setattr("vesmod.EdgeMod.spectrum_ensemble.fit_spectrum_lmfit", fake_fit)
    monkeypatch.setattr("vesmod.EdgeMod.spectrum_ensemble.validate_lmfit_result", lambda *args: None)

    result = avg._extract_kC_from_fit(lower_bound=2, upper_bound=4, lmax=700)

    assert result == 123.0
    assert calls["lmax"] == 700
    assert calls["free_sigma"] is False
    assert calls["weighted"] is False
    np.testing.assert_array_equal(calls["group"].modes, np.array([2, 3]))
    np.testing.assert_allclose(calls["group"].avg_amps2, np.array([30.0, 45.0]))
    np.testing.assert_allclose(calls["group"].avg_amps2_ste, np.array([10.0, 15.0]))

def test_kC_property_returns_value_from_extract_kC_from_fit(monkeypatch):
    """Test that the kC property delegates to _extract_kC_from_fit."""
    avg = SpectrumEnsemble()

    monkeypatch.setattr(avg, "_extract_kC_from_fit", lambda: 321.0)

    assert avg.kC == 321.0


def test_extract_kc_from_fit_accepts_free_sigma_and_records_reduced_tension(monkeypatch):
    """The public fit API honors configuration and retains fit diagnostics."""
    from types import SimpleNamespace

    avg = SpectrumEnsemble()
    avg.add_spectrum([10.0, 20.0, 30.0, 40.0], [1, 2, 3, 4], 20.0)
    avg.add_spectrum([20.0, 40.0, 60.0, 80.0], [1, 2, 3, 4], 22.0)
    config = SpectrumFitConfig(lmax=700, free_sigma=True, lower_bound=2, upper_bound=4)
    calls = {}

    def fake_fit(group, lmax, free_sigma, weighted=False):
        calls.update(group=group, lmax=lmax, free_sigma=free_sigma, weighted=weighted)
        return SimpleNamespace(best_values={"kC": 123.0, "sigma": 4.5}, chisqr=2.5, redchi=2.5)

    monkeypatch.setattr("vesmod.EdgeMod.spectrum_ensemble.fit_spectrum_lmfit", fake_fit)
    monkeypatch.setattr("vesmod.EdgeMod.spectrum_ensemble.validate_lmfit_result", lambda *args: None)

    fit = avg.extract_kc_from_fit(config, weight_by_replica_sem=True)

    assert isinstance(fit, EnsembleFit)
    assert fit.kC == 123.0
    assert fit.reduced_sigma == 4.5
    assert fit.config is config
    assert fit.weight_by_replica_sem is True
    assert fit.chisqr == pytest.approx(2.5)
    assert fit.redchi == pytest.approx(2.5)
    assert avg.fit_results == [fit]
    assert calls["lmax"] == 700
    assert calls["free_sigma"] is True
    assert calls["weighted"] is True
    np.testing.assert_array_equal(calls["group"].modes, np.array([2, 3]))
    np.testing.assert_allclose(calls["group"].avg_amps2_ste, np.array([10.0, 15.0]))

def test_extract_kc_from_fit_defaults_to_legacy_fixed_sigma(monkeypatch):
    """An omitted config preserves historical fixed-sigma unweighted fitting."""
    from types import SimpleNamespace

    avg = SpectrumEnsemble()
    avg.add_spectrum([10., 20., 30., 40., 50., 60., 70., 80.], range(1, 9), 20.)
    calls = {}

    def fake_fit(group, lmax, free_sigma, weighted=False):
        calls["weighted"] = weighted
        return SimpleNamespace(best_values={"kC": 123.0, "sigma": 0.0}, chisqr=1.0, redchi=1.0)

    monkeypatch.setattr("vesmod.EdgeMod.spectrum_ensemble.fit_spectrum_lmfit", fake_fit)
    monkeypatch.setattr("vesmod.EdgeMod.spectrum_ensemble.validate_lmfit_result", lambda *args: None)

    fit = avg.extract_kc_from_fit()

    assert fit.kC == 123.0
    assert fit.reduced_sigma == 0.0
    assert fit.config.free_sigma is False
    assert fit.weight_by_replica_sem is False
    assert calls["weighted"] is False

def test_extract_kc_from_fit_rejects_non_configuration_value():
    """The public fit API has the same configuration type contract as Spectrum."""
    avg = SpectrumEnsemble()

    with pytest.raises(TypeError, match="config must be a SpectrumFitConfig or None"):
        avg.extract_kc_from_fit("free sigma")


def test_ensemble_fit_serializes_reduced_sigma():
    """Ensemble fits expose reduced tension without claiming SI units."""
    config = SpectrumFitConfig(free_sigma=True)
    fit = EnsembleFit(kC=12.0, reduced_sigma=3.5, config=config)

    assert fit.to_dict() == {
        "kC": 12.0,
        "reduced_sigma": 3.5,
        "config": config.to_dict(),
        "weight_by_replica_sem": False,
        "chisqr": None,
        "redchi": None,
        "radii": None,
    }



def test_sem_weighting_requires_at_least_two_replicas():
    """SEM weighting rejects ensembles that cannot estimate between-replica SEM."""
    avg = SpectrumEnsemble()
    avg.add_spectrum([1.0, 2.0, 3.0], [2, 3, 4], 20.0)

    with pytest.raises(ValueError, match="At least two replica spectra"):
        avg.extract_kc_from_fit(
            SpectrumFitConfig(lower_bound=2, upper_bound=4, free_sigma=False),
            weight_by_replica_sem=True,
        )


def test_sem_weighting_option_must_be_boolean():
    """The weighting option has an explicit boolean contract."""
    avg = SpectrumEnsemble()

    with pytest.raises(TypeError, match="weight_by_replica_sem must be a bool"):
        avg.extract_kc_from_fit(weight_by_replica_sem="yes")
