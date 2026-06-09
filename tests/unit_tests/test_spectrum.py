#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/test_spectrum.py
import json

import numpy as np
import pytest

from vesmod.EdgeMode import Spectrum, MiniSpectrum


def test_init_from_npy_calculates_r0_avg_amps2_and_integer_modes(tmp_path):
    """Test that a constant npy trajectory gives r0 equal to the constant radius, unit q=0 power, zero nonzero-mode power, and FFT integer modes."""
    r_vals = np.full((3, 4), 2.0)
    infile = tmp_path / "edges.npy"
    np.save(infile, r_vals)

    spectrum = Spectrum(infile)

    assert spectrum.r0 == pytest.approx(2.0)
    np.testing.assert_allclose(spectrum.avg_amps2, np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(spectrum.modes, np.array([0, 1, -2, -1]))
    assert spectrum.kC is None
    assert spectrum.surface_tension is None


def test_init_accepts_string_path_to_npy_file(tmp_path):
    """Test that edges_over_time may be supplied as a string path to an existing .npy file."""
    infile = tmp_path / "edges.npy"
    np.save(infile, np.full((2, 4), 3.0))

    spectrum = Spectrum(str(infile))

    assert spectrum.r0 == pytest.approx(3.0)


def test_init_applies_frame_cutoff_before_calculating_r0(tmp_path):
    """Test that frame_cutoff keeps only the first requested frames before computing the average vesicle radius."""
    r_vals = np.array(
        [
            [1.0, 1.0],
            [3.0, 3.0],
            [100.0, 100.0],
        ]
    )
    infile = tmp_path / "edges.npy"
    np.save(infile, r_vals)

    spectrum = Spectrum(infile, frame_cutoff=2)

    assert spectrum.r0 == pytest.approx(2.0)


def test_init_rejects_missing_file(tmp_path):
    """Test that constructing Spectrum from a path raises ValueError when the file does not exist."""
    with pytest.raises(ValueError, match="does not appear to be a file"):
        Spectrum(tmp_path / "missing.npy")


def test_init_rejects_non_npy_file(tmp_path):
    """Test that constructing Spectrum from an existing file raises ValueError when the suffix is not .npy."""
    infile = tmp_path / "edges.txt"
    infile.write_text("not an npy file")

    with pytest.raises(ValueError, match="must end in .npy"):
        Spectrum(infile)


def test_init_rejects_invalid_edges_over_time_type():
    """Test that edges_over_time must be a path-like object or VesicleVideo instance, not an arbitrary object."""
    with pytest.raises(TypeError, match="str, pathlib Path, or VesicleVideo"):
        Spectrum(object())


def test_init_rejects_non_integer_frame_cutoff(tmp_path):
    """Test that frame_cutoff must be None or an integer and rejects floating-point values."""
    infile = tmp_path / "edges.npy"
    np.save(infile, np.full((2, 4), 1.0))

    with pytest.raises(TypeError, match="frame_cutoff must either be None or an int"):
        Spectrum(infile, frame_cutoff=1.5)


def test_init_rejects_nonpositive_frame_cutoff(tmp_path):
    """Test that frame_cutoff must be positive and rejects zero."""
    infile = tmp_path / "edges.npy"
    np.save(infile, np.full((2, 4), 1.0))

    with pytest.raises(ValueError, match="frame_cutoff must be a positive int"):
        Spectrum(infile, frame_cutoff=0)


def test_calc_avg_sq_amplitudes_matches_manual_fft_normalization(tmp_path):
    """Test that average squared amplitudes are computed from fft(r) divided by r0*n_samples and averaged over frames."""
    r_vals = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
        ]
    )
    infile = tmp_path / "edges.npy"
    np.save(infile, r_vals)
    spectrum = Spectrum(infile)

    n_samples = r_vals.shape[1]
    expected_amps = np.fft.fft(r_vals, axis=1, norm="backward") / (spectrum.r0 * n_samples)
    expected_avg_amps2 = np.mean((expected_amps * expected_amps.conj()).real, axis=0)

    np.testing.assert_allclose(spectrum.avg_amps2, expected_avg_amps2)


def test_calc_integer_modes_returns_positive_then_negative_fft_modes(tmp_path):
    """Test that integer modes follow numpy FFT ordering for four samples: 0, 1, -2, -1."""
    infile = tmp_path / "edges.npy"
    np.save(infile, np.full((2, 4), 1.0))
    spectrum = Spectrum(infile)

    np.testing.assert_array_equal(spectrum._calc_integer_modes(), np.array([0, 1, -2, -1]))


def test_isolate_mode_range_includes_lower_bound_and_excludes_upper_bound():
    """Test that isolate_mode_range returns modes >= lower_bound and < upper_bound with matching amplitudes and std_amps2 set to None."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.modes = np.array([-2, -1, 0, 1, 2, 3, 4])
    spectrum.avg_amps2 = np.array([20.0, 10.0, 0.0, 1.0, 2.0, 3.0, 4.0])

    isolated = spectrum.isolate_mode_range(lower_bound=1, upper_bound=4)

    assert isinstance(isolated, MiniSpectrum)
    np.testing.assert_array_equal(isolated.modes, np.array([1, 2, 3]))
    np.testing.assert_array_equal(isolated.avg_amps2, np.array([1.0, 2.0, 3.0]))
    assert isolated.std_amps2 is None


def test_isolate_mode_range_raises_when_modes_are_missing():
    """Test that isolate_mode_range raises AttributeError when the Spectrum object has no modes to filter."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.modes = None
    spectrum.avg_amps2 = np.array([1.0])

    with pytest.raises(AttributeError, match="There are no modes"):
        spectrum.isolate_mode_range(1, 3)


def test_extract_kc_from_fit_uses_mode_range_and_saves_fit_results(monkeypatch):
    """Test that extract_kc_from_fit fits only the requested mode range, converts reduced tension, and saves kC and surface_tension."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = 10.0
    spectrum.modes = np.array([0, 1, 2, 3, 4, 5])
    spectrum.avg_amps2 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    spectrum.kC = None
    spectrum.surface_tension = None

    calls = {}

    def fake_fit_spectrum_to_theory_lmfit(fitting_range, lmax, free_sigma):
        calls["modes"] = fitting_range.modes.copy()
        calls["avg_amps2"] = fitting_range.avg_amps2.copy()
        calls["lmax"] = lmax
        calls["free_sigma"] = free_sigma
        return 22.0, 3.5

    def fake_calc_tension_from_reduced_tension(r0, reduced_tension, kc, temperature):
        calls["tension_args"] = (r0, reduced_tension, kc, temperature)
        return 9.9

    monkeypatch.setattr("vesmod.spectrum.fit_spectrum_to_theory_lmfit", fake_fit_spectrum_to_theory_lmfit)
    monkeypatch.setattr("vesmod.spectrum.calc_tension_from_reduced_tension", fake_calc_tension_from_reduced_tension)

    kc, surface_tension = spectrum.extract_kc_from_fit(
        lower_bound=2,
        upper_bound=5,
        lmax=123,
        free_sigma=True,
        temperature=310.0,
    )

    np.testing.assert_array_equal(calls["modes"], np.array([2, 3, 4]))
    np.testing.assert_array_equal(calls["avg_amps2"], np.array([0.2, 0.3, 0.4]))
    assert calls["lmax"] == 123
    assert calls["free_sigma"] is True
    assert calls["tension_args"] == (10.0, 3.5, 22.0, 310.0)
    assert kc == 22.0
    assert surface_tension == 9.9
    assert spectrum.kC == 22.0
    assert spectrum.surface_tension == 9.9


def test_to_dict_includes_arrays_when_requested():
    """Test that _to_dict includes scalar attributes and converts numpy arrays to lists when include_arrays is True."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = np.float64(10.0)
    spectrum.kC = np.float64(20.0)
    spectrum.surface_tension = np.float64(1e-7)
    spectrum.modes = np.array([0, 1, -1])
    spectrum.avg_amps2 = np.array([1.0, 0.1, 0.1])

    data = spectrum._to_dict(include_arrays=True)

    assert data == {
        "r0": 10.0,
        "kC": 20.0,
        "surface_tension": 1e-7,
        "modes": [0, 1, -1],
        "avg_amps2": [1.0, 0.1, 0.1],
    }


def test_to_dict_excludes_arrays_when_requested():
    """Test that _to_dict omits modes and avg_amps2 when include_arrays is False."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = 10.0
    spectrum.kC = None
    spectrum.surface_tension = None
    spectrum.modes = np.array([0])
    spectrum.avg_amps2 = np.array([1.0])

    data = spectrum._to_dict(include_arrays=False)

    assert data == {
        "r0": 10.0,
        "kC": None,
        "surface_tension": None,
    }


def test_to_json_writes_json_suffix_and_serialized_spectrum_data(tmp_path):
    """Test that to_json forces a .json suffix and writes the serialized Spectrum dictionary to disk."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = 10.0
    spectrum.kC = 20.0
    spectrum.surface_tension = 1e-7
    spectrum.modes = np.array([0, 1])
    spectrum.avg_amps2 = np.array([1.0, 0.1])

    outfile = tmp_path / "spectrum_output.txt"
    spectrum.to_json(outfile, include_arrays=True, indent=2)

    json_file = tmp_path / "spectrum_output.json"
    assert json_file.is_file()
    assert json.loads(json_file.read_text()) == {
        "r0": 10.0,
        "kC": 20.0,
        "surface_tension": 1e-7,
        "modes": [0, 1],
        "avg_amps2": [1.0, 0.1],
    }
