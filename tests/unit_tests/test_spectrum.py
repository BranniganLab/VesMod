#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/test_spectrum.py
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vesmod.EdgeMod import Spectrum, SpectrumFitConfig
from vesmod.EdgeMod.spectrum_utils import MiniSpectrum
from vesmod.VesEdge import (
    EdgeDetection,
    EdgeDetectionFailure,
    EdgeExtractionConfig,
    EdgeQCConfig,
    QCFlag,
    VesicleEdges,
)
from vesmod.VesEdge.models import ImageContour


def test_init_from_npy_calculates_r0_avg_amps2_and_integer_modes(tmp_path):
    """Test that a constant npy trajectory gives the expected spectrum."""
    r_vals = np.full((3, 4), 2.0)
    infile = tmp_path / "edges.npy"
    np.save(infile, r_vals)

    spectrum = Spectrum(infile)

    assert spectrum.r0 == pytest.approx(2.0)
    np.testing.assert_allclose(
        spectrum.avg_amps2,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        spectrum.modes,
        np.array([0, 1, -2, -1]),
    )
    assert spectrum.kC is None
    assert spectrum.surface_tension is None


def test_init_accepts_string_path_to_npy_file(tmp_path):
    """Test that edges_over_time may be supplied as a string path."""
    infile = tmp_path / "edges.npy"
    np.save(infile, np.full((2, 4), 3.0))
    spectrum = Spectrum(str(infile))
    assert spectrum.r0 == pytest.approx(3.0)


def test_init_from_vesicle_edges_uses_only_accepted_detections():
    """Test that VesicleEdges input contributes only accepted radii."""
    extraction_config = EdgeExtractionConfig(
        pixels_per_micron=2.0,
        n_angular_samples=4,
    )
    qc_config = EdgeQCConfig(
        curvature_threshold=10.0,
        enable_curvature_qc=False,
        enable_area_qc=False,
    )

    accepted_contour = ImageContour((0.0, 0.0), np.full(4, 4.0))
    accepted = EdgeDetection(accepted_contour, accepted_contour)
    rejected_contour = ImageContour((0.0, 0.0), np.full(4, 18.0))
    rejected = EdgeDetection(rejected_contour, rejected_contour)
    edges = VesicleEdges(
        extraction_config,
        [accepted, rejected, EdgeDetectionFailure("failure")],
    )
    edges.run_qc(qc_config)
    rejected.qc.flags.add(QCFlag.CURVATURE)

    spectrum = Spectrum(edges)

    assert spectrum.r0 == pytest.approx(2.0)
    np.testing.assert_allclose(
        spectrum.avg_amps2,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_init_from_vesicle_edges_requires_qc():
    """Test that Spectrum rejects extracted edges before QC has run."""
    contour = ImageContour((0.0, 0.0), np.full(4, 2.0))
    edges = VesicleEdges(
        EdgeExtractionConfig(1.0, 4),
        [EdgeDetection(contour, contour)],
    )

    with pytest.raises(ValueError, match="Quality control has not been run"):
        Spectrum(edges)


def test_init_applies_frame_cutoff_before_calculating_r0(tmp_path):
    """Test that frame_cutoff keeps only the requested first frames."""
    r_vals = np.array([[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]])
    infile = tmp_path / "edges.npy"
    np.save(infile, r_vals)
    spectrum = Spectrum(infile, frame_cutoff=2)
    assert spectrum.r0 == pytest.approx(2.0)


def test_init_rejects_missing_file(tmp_path):
    """Test that a missing input path is rejected."""
    with pytest.raises(ValueError, match="does not appear to be a file"):
        Spectrum(tmp_path / "missing.npy")


def test_init_rejects_non_npy_file(tmp_path):
    """Test that file inputs must be .npy files."""
    infile = tmp_path / "edges.txt"
    infile.write_text("not an npy file")
    with pytest.raises(ValueError, match=r"must end in \.npy"):
        Spectrum(infile)


def test_init_rejects_invalid_edges_over_time_type():
    """Test that arbitrary objects are rejected as Spectrum input."""
    with pytest.raises(TypeError, match="str, pathlib Path, or VesicleEdges"):
        Spectrum(object())


def test_init_rejects_non_integer_frame_cutoff(tmp_path):
    """Test that frame_cutoff must be None or an integer."""
    infile = tmp_path / "edges.npy"
    np.save(infile, np.full((2, 4), 1.0))
    with pytest.raises(
        TypeError,
        match="frame_cutoff must either be None or an int",
    ):
        Spectrum(infile, frame_cutoff=1.5)


def test_init_rejects_nonpositive_frame_cutoff(tmp_path):
    """Test that frame_cutoff must be positive."""
    infile = tmp_path / "edges.npy"
    np.save(infile, np.full((2, 4), 1.0))
    with pytest.raises(ValueError, match="frame_cutoff must be a positive int"):
        Spectrum(infile, frame_cutoff=0)


def test_calc_avg_sq_amplitudes_matches_manual_fft_normalization(tmp_path):
    """Test FFT normalization and averaging against a manual calculation."""
    r_vals = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    infile = tmp_path / "edges.npy"
    np.save(infile, r_vals)
    spectrum = Spectrum(infile)

    n_samples = r_vals.shape[1]
    expected_amps = np.fft.fft(
        r_vals,
        axis=1,
        norm="backward",
    ) / (spectrum.r0 * n_samples)
    expected_avg_amps2 = np.mean(
        (expected_amps * expected_amps.conj()).real,
        axis=0,
    )
    np.testing.assert_allclose(spectrum.avg_amps2, expected_avg_amps2)


def test_calc_integer_modes_returns_positive_then_negative_fft_modes(tmp_path):
    """Test integer modes follow NumPy FFT ordering."""
    infile = tmp_path / "edges.npy"
    np.save(infile, np.full((2, 4), 1.0))
    spectrum = Spectrum(infile)
    np.testing.assert_array_equal(
        spectrum._calc_integer_modes(),
        np.array([0, 1, -2, -1]),
    )


def test_isolate_mode_range_includes_lower_bound_and_excludes_upper_bound():
    """Test lower-inclusive, upper-exclusive mode selection."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.modes = np.array([-2, -1, 0, 1, 2, 3, 4])
    spectrum.avg_amps2 = np.array([20.0, 10.0, 0.0, 1.0, 2.0, 3.0, 4.0])

    isolated = spectrum.isolate_mode_range(lower_bound=1, upper_bound=4)

    assert isinstance(isolated, MiniSpectrum)
    np.testing.assert_array_equal(isolated.modes, np.array([1, 2, 3]))
    np.testing.assert_array_equal(isolated.avg_amps2, np.array([1.0, 2.0, 3.0]))
    assert isolated.std_amps2 is None


def test_isolate_mode_range_raises_when_modes_are_missing():
    """Test that mode selection fails clearly when modes are unavailable."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.modes = None
    spectrum.avg_amps2 = np.array([1.0])
    with pytest.raises(AttributeError, match="There are no modes"):
        spectrum.isolate_mode_range(1, 3)


def test_extract_kc_from_fit_uses_config_and_saves_fit_results(monkeypatch):
    """Test fitting uses the configured modes and physical-fit settings."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = 10.0
    spectrum.modes = np.array([0, 1, 2, 3, 4, 5])
    spectrum.avg_amps2 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    spectrum.kC = None
    spectrum.surface_tension = None
    spectrum.fit_result = None
    spectrum.fit_results = []
    calls = {}

    def fake_fit_spectrum_lmfit(fitting_range, lmax, free_sigma):
        calls["modes"] = fitting_range.modes.copy()
        calls["avg_amps2"] = fitting_range.avg_amps2.copy()
        calls["lmax"] = lmax
        calls["free_sigma"] = free_sigma
        return SimpleNamespace(best_values={"kC": 22.0, "sigma": 3.5})

    def fake_calc_tension_from_reduced_tension(
        r0,
        reduced_tension,
        kc,
        temperature,
    ):
        calls["tension_args"] = (r0, reduced_tension, kc, temperature)
        return 9.9

    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.fit_spectrum_lmfit",
        fake_fit_spectrum_lmfit,
    )
    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.validate_lmfit_result",
        lambda result, fitting_range, free_sigma: None,
    )
    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.calc_tension_from_reduced_tension",
        fake_calc_tension_from_reduced_tension,
    )
    config = SpectrumFitConfig(
        lower_bound=2,
        upper_bound=5,
        lmax=123,
        free_sigma=True,
        temperature=310.0,
    )

    kc, surface_tension = spectrum.extract_kc_from_fit(config)

    np.testing.assert_array_equal(calls["modes"], np.array([2, 3, 4]))
    np.testing.assert_array_equal(
        calls["avg_amps2"],
        np.array([0.2, 0.3, 0.4]),
    )
    assert calls["lmax"] == 123
    assert calls["free_sigma"] is True
    assert calls["tension_args"] == (10.0, 3.5, 22.0, 310.0)
    assert kc == 22.0
    assert surface_tension == 9.9
    assert spectrum.kC == 22.0
    assert spectrum.surface_tension == 9.9
    assert spectrum.fit_results[0].config is config


def test_extract_kc_from_fit_uses_default_config(monkeypatch):
    """Test omitted config preserves the historical fixed q=3--7 fit."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = 10.0
    spectrum.modes = np.arange(0, 9)
    spectrum.avg_amps2 = np.ones(9)
    spectrum.kC = None
    spectrum.surface_tension = None
    spectrum.fit_result = None
    spectrum.fit_results = []
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
        lambda *args: 1.0,
    )

    fit = spectrum.extract_kc_from_fit()

    np.testing.assert_array_equal(calls["modes"], np.arange(3, 8))
    assert calls["lmax"] == 500
    assert calls["free_sigma"] is True
    assert fit.config.lower_bound == 3
    assert fit.config.upper_bound == 8


def test_extract_kc_from_fit_rejects_empty_configured_range(monkeypatch):
    """Test a fixed config containing no spectrum modes fails before fitting."""
    spectrum = Spectrum.__new__(Spectrum)
    spectrum.r0 = 10.0
    spectrum.modes = np.arange(0, 6)
    spectrum.avg_amps2 = np.ones(6)
    spectrum.kC = None
    spectrum.surface_tension = None
    spectrum.fit_result = None
    spectrum.fit_results = []

    monkeypatch.setattr(
        "vesmod.EdgeMod.spectrum.fit_spectrum_lmfit",
        lambda *args: pytest.fail("Physical fitter should not run"),
    )

    with pytest.raises(ValueError, match="contains no spectrum modes"):
        spectrum.extract_kc_from_fit(
            SpectrumFitConfig(lower_bound=7, upper_bound=9)
        )


def test_to_dict_includes_arrays_when_requested():
    """Test serialization includes arrays when requested."""
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
    """Test serialization can omit arrays when requested."""
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
    """Test JSON export uses the requested Spectrum data."""
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
