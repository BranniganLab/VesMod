"""Physical and integration tests for finite camera exposure fitting."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.constants import Boltzmann
from scipy.integrate import quad
from vesmod.EdgeMod import Spectrum, SpectrumFitConfig, SpectrumEnsemble
from vesmod.EdgeMod.spectrum_utils import (
    HSS97, HSS97_with_camera_integration_time, MiniSpectrum, exposure_power_factor, fit_spectrum_lmfit,
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
    assert HSS97_with_camera_integration_time(q, 25, 2, 40, config=SpectrumFitConfig(), radii=None) == HSS97(q, 25, 2, 40)


def test_single_spherical_mode_si_units():
    """Check l=3 rate independently with explicit SI conversions."""
    config = SpectrumFitConfig(viscosity_in=.00102, viscosity_out=.00097, exposure_time=0.030)
    rate = (25 * Boltzmann * 295 / (0.00097 * (5e-6)**3)
            * (2 * 3 * 4 * 5 * (12 + 2))
            / (4 * 27 + 6 * 9 - 1 + (2 * 27 + 3 * 9 - 5) * (1.02 / .97 - 1)))
    predicted = HSS97_with_camera_integration_time([3], 25, 2, 3, config=config, radii=[5])[0]
    assert predicted / HSS97([3], 25, 2, 3)[0] == pytest.approx(exposure_power_factor(.030 * rate))


def test_radius_and_exposure_dependence():
    """Smaller radii and longer exposures suppress more contour power."""
    config = SpectrumFitConfig(viscosity_in=.00102, viscosity_out=.00097, exposure_time=.030)
    small = np.array(HSS97_with_camera_integration_time([3, 5, 7], 25, 0, 50, config=config, radii=[5]))
    large = np.array(HSS97_with_camera_integration_time([3, 5, 7], 25, 0, 50, config=config, radii=[10]))
    long = np.array(HSS97_with_camera_integration_time([3, 5, 7], 25, 0, 50, config=replace(config, exposure_time=.060), radii=[5]))
    assert np.all(long < small)
    assert np.all(small < large)
    assert np.all(large < HSS97([3, 5, 7], 25, 0, 50))


@pytest.mark.parametrize('free_sigma', [False, True])
def test_fit_recovers_synthetic_parameters(free_sigma):
    """Recover known parameters from a camera-averaged synthetic spectrum."""
    config = SpectrumFitConfig(viscosity_in=.00102, viscosity_out=.00097, lmax=50, upper_bound=11, exposure_time=.030, free_sigma=free_sigma)
    q = np.arange(3, 11)
    sigma = 4 if free_sigma else 0
    measured = np.array(HSS97_with_camera_integration_time(q, 25, sigma, 50, config=config, radii=[5]))
    result = fit_spectrum_lmfit(MiniSpectrum(q, measured, measured * .01), 50,
                                free_sigma, weighted=True, config=config, radii=[5])
    assert result.best_values['kC'] == pytest.approx(25, rel=1e-5)
    assert result.best_values['sigma'] == pytest.approx(sigma, abs=1e-5)
    np.testing.assert_allclose(result.eval(q=q), measured, rtol=1e-6)


def test_spectrum_and_ensemble_use_individual_radii():
    """Exercise public fitting and retain settings and replica radii."""
    config = SpectrumFitConfig(viscosity_in=.00102, viscosity_out=.00097, lmax=50, exposure_time=.030, free_sigma=False)
    ensemble = SpectrumEnsemble()
    predictions = []
    for radius in [5, 15]:
        spectrum = Spectrum.from_radii(np.full((2, 32), radius, dtype=float))
        spectrum.avg_amps2[3:8] = HSS97_with_camera_integration_time(range(3, 8), 25, 0, 50, config=config, radii=[radius])
        fit = spectrum.extract_kc_from_fit(config)
        assert fit.kC == pytest.approx(25, rel=1e-5)
        assert fit.to_dict()['config']['exposure_time'] == .030
        ensemble.add_spectrum(spectrum.avg_amps2, spectrum.modes, fit.kC, r0=radius)
        predictions.append(spectrum.avg_amps2[3:8])
    np.testing.assert_allclose(HSS97_with_camera_integration_time(range(3, 8), 25, 0, 50, config=config, radii=[5, 15]),
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
    config = SpectrumFitConfig(viscosity_in=.00102, viscosity_out=.00097, lmax=50, free_sigma=False, exposure_time=.030)
    q = np.arange(3, 8)
    measured = np.array(HSS97_with_camera_integration_time(q, 25, 0, 50, config=config, radii=[5]))
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
    config = SpectrumFitConfig(viscosity_in=.00102, viscosity_out=.00097, exposure_time=.030)
    with pytest.raises(ValueError, match='radii are required'):
        HSS97_with_camera_integration_time([3], 25, 0, 50, config=config, radii=None)
    with pytest.raises(ValueError, match='stable spherical modes'):
        HSS97_with_camera_integration_time([3], 25, -6, 50, config=config, radii=[5])


@pytest.mark.parametrize("radii", [[10], [5, 10, 15]])
def test_camera_fit_rejects_wrong_replica_radius_count(radii):
    config = SpectrumFitConfig(viscosity_in=.00102, viscosity_out=.00097,
                               exposure_time=.030)
    group = MiniSpectrum(np.arange(3, 8), np.ones(5), None)
    with pytest.raises(ValueError, match="expected 2 radii, received"):
        fit_spectrum_lmfit(group, 50, config=config, radii=radii,
                          expected_replica_count=2)


@pytest.mark.parametrize("radii", [[5], [5, 15]])
def test_camera_fit_accepts_matching_replica_radius_count(radii):
    config = SpectrumFitConfig(viscosity_in=.00102, viscosity_out=.00097,
                               exposure_time=.030, lmax=50)
    q = np.arange(3, 8)
    power = HSS97_with_camera_integration_time(q, 25, 0, 50,
                                              config=config, radii=radii)
    result = fit_spectrum_lmfit(MiniSpectrum(q, np.array(power), None), 50,
                               config=config, radii=radii,
                               expected_replica_count=len(radii))
    assert result.best_values["kC"] == pytest.approx(25, rel=1e-5)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_camera_fit_rejects_invalid_expected_replica_count(count):
    config = SpectrumFitConfig(viscosity_in=.00102, viscosity_out=.00097,
                               exposure_time=.030)
    group = MiniSpectrum(np.arange(3, 8), np.ones(5), None)
    with pytest.raises(ValueError, match="positive integer"):
        fit_spectrum_lmfit(group, 50, config=config, radii=[5],
                          expected_replica_count=count)


@pytest.mark.parametrize("kwargs", [{}, {"viscosity_in": .001}, {"viscosity_out": .001}])
def test_exposure_requires_explicit_viscosities(kwargs):
    """Experiment-specific solvent values must be supplied by the caller."""
    with pytest.raises(ValueError, match="viscosity_in and viscosity_out are required"):
        SpectrumFitConfig(exposure_time=.030, **kwargs)


@pytest.mark.parametrize("kc_kbt", [1.0, 25.0, 499.0])
@pytest.mark.parametrize("reduced_sigma", [-5.99, 0.0, 1000.0])
@pytest.mark.parametrize("viscosities", [(.001, .001), (.00102, .00097), (.01, .0005)])
@pytest.mark.parametrize("temperature", [280.0, 310.0])
def test_implemented_relaxation_matches_full_tau_expression(
    monkeypatch, kc_kbt, reduced_sigma, viscosities, temperature,
):
    """Compare actual optimizer rates with the unfactored Faizi eq. 2.

    Capture the arguments passed by HSS97_with_camera_integration_time into exposure_power_factor:
    these are t_exp/tau_l, so dividing by exposure yields the implemented
    rates. This exercises the production calculation, rather than a copy of
    its shortcuts. q=2 and q=3 together cover every l from 2 through 500.

    The reference explicitly computes tau_l (seconds), with the entire
    polynomial numerator and denominator, independently of the production
    stiffness and rate_scale intermediates. Extended precision keeps reference
    rounding below the 5e-15 relative tolerance on float64 production values.
    """
    import vesmod.EdgeMod.spectrum_utils as utils

    radii_um = np.array([2.0, 5.0, 15.0, 50.0])
    config = SpectrumFitConfig(
        lmax=500,
        exposure_time=.030,
        temperature=temperature,
        viscosity_in=viscosities[0],
        viscosity_out=viscosities[1],
    )
    captured = []
    original_factor = utils.exposure_power_factor

    def capture_exposure_ratio(x):
        captured.append(np.array(x, copy=True))
        return original_factor(x)

    monkeypatch.setattr(utils, "exposure_power_factor", capture_exposure_ratio)
    utils.HSS97_with_camera_integration_time([2, 3], kc_kbt, reduced_sigma, 500,
                       config=config, radii=radii_um)
    assert len(captured) == 2

    # Full expression, without sharing any production relaxation helpers.
    real = np.longdouble
    kappa_joule = real(kc_kbt) * real(Boltzmann) * real(temperature)
    eta_in = real(viscosities[0])
    eta_out = real(viscosities[1])
    for q, exposure_ratio in zip([2, 3], captured):
        ell = np.arange(q, 501, 2, dtype=np.longdouble)[None, :]
        radius_m = (radii_um.astype(np.longdouble) * real('1e-6'))[:, None]
        tau_seconds = (
            eta_out * radius_m**3
            * (4 * ell**3 + 6 * ell**2 - 1
               + (2 * ell**3 + 3 * ell**2 - 5) * (eta_in / eta_out - 1))
            / (kappa_joule * (ell - 1) * ell * (ell + 1) * (ell + 2)
               * (ell * (ell + 1) + real(reduced_sigma)))
        )
        implemented_tau = config.exposure_time / exposure_ratio
        np.testing.assert_allclose(
            implemented_tau, tau_seconds, rtol=5e-15, atol=0,
            err_msg="Factored implementation differs from full spherical relaxation time",
        )
