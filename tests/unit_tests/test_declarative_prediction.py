"""Behavioral coverage of explicit model, exposure, and relaxation choices."""
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vesmod.EdgeMod import (
    Spectrum, SpectrumFitConfig, mode_relaxation_times, predict_spectrum,
    spherical_contour_spectrum,
)


def test_exposure_metadata_does_not_enable_camera_integration():
    config = SpectrumFitConfig(exposure_time=.030)
    expected = spherical_contour_spectrum([3, 5, 7], 25, 4, 50)
    assert predict_spectrum([3, 5, 7], 25, 4, 50, config=config) == expected


def test_toggle_retains_parameters_and_changes_only_exposure_averaging():
    config = SpectrumFitConfig(camera_integration=True, exposure_time=.030,
                               viscosity_in=.00102, viscosity_out=.00097)
    disabled = replace(config, camera_integration=False)
    equilibrium = spherical_contour_spectrum([3, 5, 7], 25, 4, 50)
    assert predict_spectrum([3, 5, 7], 25, 4, 50, config=disabled) == equilibrium
    assert np.all(np.array(predict_spectrum([3, 5, 7], 25, 4, 50,
                                           config=config, radii=[5])) < equilibrium)
    assert disabled.exposure_time == .030


@pytest.mark.parametrize('kwargs, message', [
    ({'model': 'unknown'}, 'model must'),
    ({'relaxation_model': 'unknown'}, 'relaxation_model must'),
    ({'camera_integration': True}, 'Positive exposure_time'),
    ({'eta_m': 0}, 'eta_m requires'),
    ({'relaxation_model': 'solvent_and_membrane'}, 'eta_m is required'),
])
def test_explicit_choices_reject_incomplete_or_incompatible_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SpectrumFitConfig(**kwargs)


def test_camera_toggle_requires_boolean():
    with pytest.raises(TypeError, match='camera_integration must be a bool'):
        SpectrumFitConfig(camera_integration=1)


def test_cli_parameters_do_not_select_correction_or_relaxation(monkeypatch):
    from vesmod.cli.edgemod_cli import build_fit_config, parse_args
    monkeypatch.setattr('sys.argv', ['edgemod', 'contours.npy', '--exposure-time', '.030'])
    config = build_fit_config(parse_args())
    assert not config.camera_integration
    assert config.relaxation_model == 'solvent_only'
    monkeypatch.setattr('sys.argv', ['edgemod', 'contours.npy', '--eta-m', '0'])
    with pytest.raises(ValueError, match='eta_m requires'):
        build_fit_config(parse_args())


def test_standalone_relaxation_times_do_not_require_camera_averaging():
    config = SpectrumFitConfig(viscosity_in=.001, viscosity_out=.001)
    membrane = replace(config, relaxation_model='solvent_and_membrane', eta_m=4.1e-9)
    solvent_times = mode_relaxation_times([2, 3, 500], 25, 4, config=config, radii=[5, 15])
    membrane_times = mode_relaxation_times([2, 3, 500], 25, 4, config=membrane, radii=[5, 15])
    assert solvent_times.shape == (2, 3)
    assert np.all(membrane_times > solvent_times)
    np.testing.assert_array_equal(solvent_times, mode_relaxation_times(
        [2, 3, 500], 25, 4, config=replace(membrane, eta_m=0), radii=[5, 15]))


def test_disabled_correction_fits_and_plots_equilibrium_with_exposure_metadata():
    config = SpectrumFitConfig(exposure_time=.030, lmax=50)
    spectrum = Spectrum.from_radii(np.full((2, 32), 5.0))
    spectrum.avg_amps2[3:8] = spherical_contour_spectrum(range(3, 8), 25, 4, 50)
    fit = spectrum.extract_kc_from_fit(config)
    assert fit.kC == pytest.approx(25, rel=1e-5)
    assert fit.to_dict()['config']['camera_integration'] is False
    figure, ax = plt.subplots()
    try:
        spectrum.plot(ax=ax)
        np.testing.assert_allclose(ax.lines[-1].get_ydata(),
                                   spectrum.avg_amps2[3:8], rtol=1e-5)
        from vesmod.EdgeMod import SpectrumPlotConfig
        with pytest.raises(ValueError, match='existing camera integration fit'):
            spectrum.plot(ax=ax, plot_config=SpectrumPlotConfig(power='corrected'))
    finally:
        plt.close(figure)


@pytest.mark.parametrize('relaxation_model, eta_m', [
    ('solvent_only', None), ('solvent_and_membrane', 4.1e-9),
])
def test_prediction_validates_inputs_once_not_each_spherical_mode(
        monkeypatch, relaxation_model, eta_m):
    """Cached ell arrays must bypass repeated public-API validation."""
    import vesmod.EdgeMod.spectrum_utils as utils
    config = SpectrumFitConfig(camera_integration=True, exposure_time=.030,
                               relaxation_model=relaxation_model, eta_m=eta_m,
                               viscosity_in=.001, viscosity_out=.001)
    calls = {'spectrum': 0, 'radii': 0}
    original_spectrum = utils._validate_spherical_inputs
    original_radii = utils._validate_camera_radii

    def validate_spectrum(q, *args):
        calls['spectrum'] += 1
        assert len(q) == 5  # q=3..7, not hundreds of cached spherical modes
        return original_spectrum(q, *args)

    def validate_radii(radii):
        calls['radii'] += 1
        return original_radii(radii)

    monkeypatch.setattr(utils, '_validate_spherical_inputs', validate_spectrum)
    monkeypatch.setattr(utils, '_validate_camera_radii', validate_radii)
    power = utils.predict_spectrum([3, 4, 5, 6, 7], 25, 4, 500,
                                   config=config, radii=[5, 15])
    assert np.all(np.isfinite(power))
    assert calls == {'spectrum': 1, 'radii': 1}
