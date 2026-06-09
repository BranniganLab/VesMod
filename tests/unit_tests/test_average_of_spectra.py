#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for AverageOfSpectra."""

import numpy as np
import pytest

from vesmod.EdgeMod.average_of_spectra import AverageOfSpectra


def test_new_average_of_spectra_is_empty():
    """Test that a new AverageOfSpectra starts with no spectra, no kC values, and no modes."""
    avg = AverageOfSpectra()

    assert len(avg) == 0
    assert avg.spectra_list == []
    assert avg.kC_list == []
    assert avg.modes is None


def test_add_spectrum_stores_first_spectrum_and_modes():
    """Test that the first added spectrum defines the stored mode array and is saved."""
    avg = AverageOfSpectra()

    avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 2, 3], kC=20.0)

    assert len(avg) == 1
    np.testing.assert_array_equal(avg.modes, np.array([1, 2, 3]))
    assert avg.spectra_list == [[1.0, 2.0, 3.0]]
    assert avg.kC_list == [20.0]


def test_add_spectrum_accepts_matching_modes_after_first_spectrum():
    """Test that additional spectra are accepted when their modes match the stored modes exactly."""
    avg = AverageOfSpectra()

    avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 2, 3], kC=20.0)
    avg.add_spectrum(avg_amps2=[2.0, 4.0, 6.0], modes=[1, 2, 3], kC=24.0)

    assert len(avg) == 2
    np.testing.assert_array_equal(avg.modes, np.array([1, 2, 3]))
    assert avg.spectra_list == [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]
    assert avg.kC_list == [20.0, 24.0]


def test_add_spectrum_rejects_nonmatching_modes():
    """Test that add_spectrum raises ValueError when a later spectrum uses different modes."""
    avg = AverageOfSpectra()

    avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 2, 3], kC=20.0)

    with pytest.raises(ValueError, match=r"\[1, 3, 5\] does not equal \[1 2 3\]"):
        avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 3, 5], kC=22.0)


def test_add_spectrum_rejects_invalid_internal_modes_state():
    """Test that add_spectrum raises TypeError if self.modes has an invalid internal type."""
    avg = AverageOfSpectra()
    avg.modes = [1, 2, 3]

    with pytest.raises(TypeError, match="self.modes must be ndarray or None"):
        avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 2, 3], kC=20.0)


def test_avg_amps2_returns_elementwise_mean_across_spectra():
    """Test that avg_amps2 returns the elementwise mean of all stored spectra."""
    avg = AverageOfSpectra()

    avg.add_spectrum(avg_amps2=[1.0, 2.0, 3.0], modes=[1, 2, 3], kC=20.0)
    avg.add_spectrum(avg_amps2=[3.0, 4.0, 5.0], modes=[1, 2, 3], kC=22.0)
    avg.add_spectrum(avg_amps2=[5.0, 6.0, 7.0], modes=[1, 2, 3], kC=24.0)

    np.testing.assert_allclose(avg.avg_amps2, np.array([3.0, 4.0, 5.0]))


def test_avg_amps2_std_returns_sample_standard_deviation_across_spectra():
    """Test that avg_amps2_std uses ddof=1 to calculate sample standard deviation."""
    avg = AverageOfSpectra()

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
    avg = AverageOfSpectra()

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
    avg = AverageOfSpectra()

    avg.add_spectrum(avg_amps2=[1.0, 2.0], modes=[1, 2], kC=20.0)
    avg.add_spectrum(avg_amps2=[2.0, 3.0], modes=[1, 2], kC=22.0)
    avg.add_spectrum(avg_amps2=[3.0, 4.0], modes=[1, 2], kC=24.0)

    expected = np.std([20.0, 22.0, 24.0], ddof=1)
    assert avg.kC_std == pytest.approx(expected)


def test_kC_ste_returns_standard_error_of_replica_kC_values():
    """Test that kC_ste divides kC_std by sqrt of the number of replica kC values."""
    avg = AverageOfSpectra()

    avg.add_spectrum(avg_amps2=[1.0, 2.0], modes=[1, 2], kC=20.0)
    avg.add_spectrum(avg_amps2=[2.0, 3.0], modes=[1, 2], kC=22.0)
    avg.add_spectrum(avg_amps2=[3.0, 4.0], modes=[1, 2], kC=24.0)

    expected = np.std([20.0, 22.0, 24.0], ddof=1) / np.sqrt(3)
    assert avg.kC_ste == pytest.approx(expected)


def test_isolate_mode_range_returns_selected_modes_and_average_amplitudes():
    """Test that _isolate_mode_range keeps modes >= lower_bound and < upper_bound."""
    avg = AverageOfSpectra()

    avg.add_spectrum(avg_amps2=[10.0, 20.0, 30.0, 40.0], modes=[1, 2, 3, 4], kC=20.0)
    avg.add_spectrum(avg_amps2=[20.0, 40.0, 60.0, 80.0], modes=[1, 2, 3, 4], kC=22.0)

    mini_spectrum = avg._isolate_mode_range(lower_bound=2, upper_bound=4)

    np.testing.assert_array_equal(mini_spectrum.modes, np.array([2, 3]))
    np.testing.assert_allclose(mini_spectrum.avg_amps2, np.array([30.0, 45.0]))
    assert mini_spectrum.std_amps2 is None


def test_isolate_mode_range_raises_error_when_no_modes_have_been_added():
    """Test that _isolate_mode_range raises AttributeError before any spectrum has set modes."""
    avg = AverageOfSpectra()

    with pytest.raises(AttributeError, match="There are no modes"):
        avg._isolate_mode_range(lower_bound=2, upper_bound=4)


def test_extract_kC_from_fit_uses_isolated_mode_range_and_returns_first_fit_value(monkeypatch):
    """Test that _extract_kC_from_fit fits the requested mode range and returns fit[0]."""
    avg = AverageOfSpectra()

    avg.add_spectrum(avg_amps2=[10.0, 20.0, 30.0, 40.0], modes=[1, 2, 3, 4], kC=20.0)
    avg.add_spectrum(avg_amps2=[20.0, 40.0, 60.0, 80.0], modes=[1, 2, 3, 4], kC=22.0)

    calls = {}

    def fake_fit_spectrum_to_theory_lmfit(fitting_range, lmax, free_sigma):
        calls["fitting_range"] = fitting_range
        calls["lmax"] = lmax
        calls["free_sigma"] = free_sigma
        return (123.0, "unused value")

    monkeypatch.setattr(
        "vesmod.EdgeMod.average_of_spectra.fit_spectrum_to_theory_lmfit",
        fake_fit_spectrum_to_theory_lmfit,
    )

    result = avg._extract_kC_from_fit(lower_bound=2, upper_bound=4, lmax=700)

    assert result == 123.0
    assert calls["lmax"] == 700
    assert calls["free_sigma"] is False
    np.testing.assert_array_equal(calls["fitting_range"].modes, np.array([2, 3]))
    np.testing.assert_allclose(calls["fitting_range"].avg_amps2, np.array([30.0, 45.0]))
    assert calls["fitting_range"].std_amps2 is None


def test_kC_property_returns_value_from_extract_kC_from_fit(monkeypatch):
    """Test that the kC property delegates to _extract_kC_from_fit."""
    avg = AverageOfSpectra()

    monkeypatch.setattr(avg, "_extract_kC_from_fit", lambda: 321.0)

    assert avg.kC == 321.0
