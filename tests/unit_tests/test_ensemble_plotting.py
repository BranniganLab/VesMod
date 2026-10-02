"""Verify ensemble figure provenance, uncertainty, and layout."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vesmod.EdgeMod import (
    EnsembleFit, EnsemblePlotConfig, SpectrumEnsemble, SpectrumFitConfig,
    plot_ensemble_spectra,
)
from vesmod.EdgeMod.spectrum_utils import HSS97_with_camera_integration_time


def make_ensemble():
    ensemble = SpectrumEnsemble()
    q = np.arange(2, 21)
    for factor, radius in [(0.8, 5), (1.2, 8)]:
        ensemble.add_spectrum(factor * 1e-3 / q**3, q, 25, r0=radius)
    return ensemble


def test_camera_curve_uses_recorded_radii_and_does_not_refit():
    ensemble = make_ensemble()
    config = SpectrumFitConfig(lmax=30, exposure_time=0.03,
                               viscosity_in=1.02e-3, viscosity_out=0.97e-3)
    fit = EnsembleFit(25, 2, config, radii=(4, 7))
    ensemble.fit_results.append(fit)
    figure, ax = ensemble.plot(label="DOPC")
    try:
        curve = next(line for line in ax.lines if line.get_label().startswith("Free sigma"))
        q = np.arange(3, 8)
        np.testing.assert_array_equal(curve.get_xdata(), q)
        np.testing.assert_allclose(curve.get_ydata(), HSS97_with_camera_integration_time(
            q, 25, 2, 30, config=config, radii=(4, 7)))
        assert ensemble.fit_results == [fit]
        assert ax.get_xscale() == ax.get_yscale() == "log"
        assert not figure.texts
    finally:
        plt.close(figure)


def test_mean_and_sem_are_equal_replica_statistics():
    ensemble = make_ensemble()
    figure, ax = ensemble.plot(fits=[])
    try:
        errorbar = ax.containers[0]
        q, mean = errorbar.lines[0].get_data()
        expected = np.mean(ensemble.spectra_list, axis=0)
        sem = np.std(ensemble.spectra_list, axis=0, ddof=1) / np.sqrt(2)
        np.testing.assert_allclose(mean, expected)
        segments = errorbar.lines[2][0].get_segments()
        np.testing.assert_allclose([segment[:, 1] for segment in segments],
                                   np.column_stack((expected - sem, expected + sem)))
        np.testing.assert_array_equal(q, ensemble.modes)
    finally:
        plt.close(figure)


def test_panels_have_shared_limits_titles_and_no_footer():
    ensemble = make_ensemble()
    ensemble.fit_results = [EnsembleFit(25, 0, SpectrumFitConfig(lmax=30, free_sigma=False)),
                            EnsembleFit(23, 2, SpectrumFitConfig(lmax=30))]
    figure, axes = plot_ensemble_spectra([ensemble] * 3,
                                        labels=["1.25% C16", "2.5% C16", "5% C16"],
                                        ncols=2, title="C16 spectra",
                                        config=EnsemblePlotConfig(power_definition="temporal_variance"))
    try:
        assert axes.shape == (2, 2)
        assert not axes[1, 1].get_visible()
        for ax in list(axes.flat)[:3]:
            assert ax.get_xlim() == (1.8, 22.5)
            assert ax.get_ylim() == (5e-8, 5e-3)
            assert "n=2" in ax.get_title() and "fixed 25.00; free 23.00" in ax.get_title()
            assert "N/m (mean radius)" in ax.get_title()
            assert ax.get_ylabel() == "Temporal-variance power"
        assert len(figure.legends) == 1
        assert [text.get_text() for text in figure.texts] == ["C16 spectra"]
    finally:
        plt.close(figure)
