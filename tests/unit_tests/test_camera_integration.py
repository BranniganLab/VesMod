"""Physical and integration tests for finite camera exposure fitting."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.constants import Boltzmann
from scipy.integrate import quad
from vesmod.EdgeMod import Spectrum, SpectrumFitConfig, SpectrumEnsemble
from vesmod.EdgeMod.spectrum_utils import (
    HSS97, HSS97_camera, MiniSpectrum, exposure_power_factor, fit_spectrum_lmfit,
)


@pytest.mark.parametrize('x', [0, 1e-9, 0.001, 0.5, 10, 100])
def test_power_factor_matches_integrated_autocorrelation(x):
    """Compare with the double exposure integral, reduced by stationarity."""
    expected = 2 * quad(lambda t: (1 - t) * np.exp(-x * t), 0, 1)[0]
    assert exposure_power_factor(x) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize('kwargs', [
    {'exposure_time': -1}, {'exposure_time': np.nan},
    {'viscosity_in': 0}, {'viscosity_out': np.inf},
])
def test_invalid_settings(kwargs):
    """Reject invalid exposure and viscosity before optimization."""
    with pytest.raises(ValueError):
        SpectrumFitConfig(**kwargs)


def test_zero_exposure_is_exactly_original_model():
    """Zero exposure requires no radii and preserves the original spectrum."""
    q = np.arange(3, 8)
    assert HSS97_camera(q, 25, 2, 40, config=SpectrumFitConfig(), radii=None) == HSS97(q, 25, 2, 40)


def test_single_spherical_mode_si_units():
    """Check l=3 rate independently with explicit SI conversions."""
    config = SpectrumFitConfig(exposure_time=0.030)
    rate = (25 * Boltzmann * 295 / (0.00097 * (5e-6)**3)
            * (2 * 3 * 4 * 5 * (12 + 2))
            / (4 * 27 + 6 * 9 - 1 + (2 * 27 + 3 * 9 - 5) * (1.02 / .97 - 1)))
    predicted = HSS97_camera([3], 25, 2, 3, config=config, radii=[5])[0]
    assert predicted / HSS97([3], 25, 2, 3)[0] == pytest.approx(exposure_power_factor(.030 * rate))


def test_radius_and_exposure_dependence():
    """Smaller radii and longer exposures suppress more contour power."""
    config = SpectrumFitConfig(exposure_time=.030)
    small = np.array(HSS97_camera([3, 5, 7], 25, 0, 50, config=config, radii=[5]))
    large = np.array(HSS97_camera([3, 5, 7], 25, 0, 50, config=config, radii=[10]))
    long = np.array(HSS97_camera([3, 5, 7], 25, 0, 50, config=replace(config, exposure_time=.060), radii=[5]))
    assert np.all(long < small)
    assert np.all(small < large)
    assert np.all(large < HSS97([3, 5, 7], 25, 0, 50))


@pytest.mark.parametrize('free_sigma', [False, True])
def test_fit_recovers_synthetic_parameters(free_sigma):
    """Recover known parameters from a camera-averaged synthetic spectrum."""
    config = SpectrumFitConfig(lmax=50, upper_bound=11, exposure_time=.030, free_sigma=free_sigma)
    q = np.arange(3, 11)
    sigma = 4 if free_sigma else 0
    measured = np.array(HSS97_camera(q, 25, sigma, 50, config=config, radii=[5]))
    result = fit_spectrum_lmfit(MiniSpectrum(q, measured, measured * .01), 50,
                                free_sigma, weighted=True, config=config, radii=[5])
    assert result.best_values['kC'] == pytest.approx(25, rel=1e-5)
    assert result.best_values['sigma'] == pytest.approx(sigma, abs=1e-5)
    np.testing.assert_allclose(result.eval(q=q), measured, rtol=1e-6)


def test_spectrum_and_ensemble_use_individual_radii():
    """Exercise public fitting and retain settings and replica radii."""
    config = SpectrumFitConfig(lmax=50, exposure_time=.030, free_sigma=False)
    ensemble = SpectrumEnsemble()
    predictions = []
    for radius in [5, 15]:
        spectrum = Spectrum.from_radii(np.full((2, 32), radius, dtype=float))
        spectrum.avg_amps2[3:8] = HSS97_camera(range(3, 8), 25, 0, 50, config=config, radii=[radius])
        fit = spectrum.extract_kc_from_fit(config)
        assert fit.kC == pytest.approx(25, rel=1e-5)
        assert fit.to_dict()['config']['exposure_time'] == .030
        ensemble.add_spectrum(spectrum.avg_amps2, spectrum.modes, fit.kC, r0=radius)
        predictions.append(spectrum.avg_amps2[3:8])
    np.testing.assert_allclose(HSS97_camera(range(3, 8), 25, 0, 50, config=config, radii=[5, 15]),
                               np.mean(predictions, axis=0))
    fit = ensemble.extract_kc_from_fit(config, weight_by_replica_sem=True)
    assert fit.kC == pytest.approx(25, rel=1e-5)
    assert fit.to_dict()['radii'] == [5, 15]
    ensemble.add_spectrum(spectrum.avg_amps2, spectrum.modes, 25)
    with pytest.raises(ValueError, match='r0 for every replica'):
        ensemble.extract_kc_from_fit(config)


def test_cli_camera_settings(monkeypatch):
    """The CLI passes exposure and SI viscosities into scientific config."""
    from vesmod.cli.edgemod_cli import parse_args, build_fit_config
    monkeypatch.setattr('sys.argv', ['edgemod', 'contours.npy', '--exposure-time', '.030',
                                    '--viscosity-in', '.00102', '--viscosity-out', '.00097'])
    config = build_fit_config(parse_args())
    assert config.exposure_time == .030
    assert config.viscosity_in == .00102
    assert config.viscosity_out == .00097


def test_corrected_plot_uses_corrected_prediction():
    """The plotted fit line follows the fitted exposure model."""
    import matplotlib.pyplot as plt
    from vesmod.EdgeMod.spectrum_plotting import SpectrumPlotData, plot_spectrum
    config = SpectrumFitConfig(lmax=50, free_sigma=False, exposure_time=.030)
    q = np.arange(3, 8)
    measured = np.array(HSS97_camera(q, 25, 0, 50, config=config, radii=[5]))
    result = fit_spectrum_lmfit(MiniSpectrum(q, measured, None), 50,
                                config=config, radii=[5])
    data = SpectrumPlotData(modes=q, avg_amps2=measured, fit_result=result,
                            lower_bound=3, upper_bound=8, lmax=50)
    figure, ax = plt.subplots()
    plot_spectrum(data, ax=ax, color="C0")
    np.testing.assert_allclose(ax.lines[-1].get_ydata(), measured, rtol=1e-6)
    plt.close(figure)


def test_camera_requires_radius_and_stable_modes():
    """Fail clearly when required physical inputs are unavailable."""
    config = SpectrumFitConfig(exposure_time=.030)
    with pytest.raises(ValueError, match='radii are required'):
        HSS97_camera([3], 25, 0, 50, config=config, radii=None)
    with pytest.raises(ValueError, match='stable spherical modes'):
        HSS97_camera([3], 25, -6, 50, config=config, radii=[5])
