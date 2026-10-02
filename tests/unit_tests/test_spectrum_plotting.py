"""Unit tests for EdgeMod spectrum plotting."""

from types import SimpleNamespace

import numpy as np
import pytest
import matplotlib.pyplot as plt

from vesmod.EdgeMod import Spectrum, SpectrumFitConfig

from vesmod.EdgeMod.spectrum_plotting import (
    SpectrumPlotData,
    SpectrumPlotConfig,
    plot_spectrum,
    save_spectrum_fit_diagnostic,
)
from vesmod.EdgeMod.spectrum_utils import HSS97


def test_save_spectrum_fit_diagnostic_creates_png_for_rejected_fit(tmp_path):
    """Test a diagnostic image is available when fit validation fails."""
    modes = np.arange(2, 10)
    measured = np.asarray(HSS97(modes, kC=25.0, sigma=2.0, lmax=30))
    fit_result = SimpleNamespace(
        best_values={"kC": 25.0, "sigma": 2.0},
        eval=lambda q: HSS97(q, kC=25.0, sigma=2.0, lmax=30),
        params={
            "kC": SimpleNamespace(value=25.0, stderr=30.0),
            "sigma": SimpleNamespace(value=2.0, stderr=4.0),
        },
    )
    output_path = tmp_path / "sample.spectrum_diagnostic.png"

    save_spectrum_fit_diagnostic(
        SpectrumPlotData(
            modes=modes,
            avg_amps2=measured,
            fit_result=fit_result,
            lower_bound=3,
            upper_bound=8,
            lmax=30,
            validation_error="poorly constrained kC",
        ),
        output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_spectrum_uses_same_color_for_data_and_fit_and_accepts_axis():
    """Test a spectrum can compose into an existing multi-panel axis."""
    import matplotlib.pyplot as plt

    modes = np.arange(2, 10)
    measured = np.asarray(HSS97(modes, kC=25.0, sigma=2.0, lmax=30))
    fit_result = SimpleNamespace(
        best_values={"kC": 25.0, "sigma": 2.0},
        eval=lambda q: HSS97(q, kC=25.0, sigma=2.0, lmax=30),
        params={
            "kC": SimpleNamespace(value=25.0, stderr=None),
            "sigma": SimpleNamespace(value=2.0, stderr=None),
        },
    )
    figure, axis = plt.subplots()
    result = plot_spectrum(
        SpectrumPlotData(
            modes=modes,
            avg_amps2=measured,
            fit_result=fit_result,
            lower_bound=3,
            upper_bound=8,
            lmax=30,
        ),
        ax=axis,
        color="purple",
        label="block 1",
        config=SpectrumPlotConfig(fitting_region="strip"),
    )

    assert result.ax is axis
    assert result.nonfit_artist.get_color() == "purple"
    assert result.fit_data_artist.get_color() == "purple"
    assert result.fit_artist.get_color() == "purple"
    assert result.fitting_region_artist is not None
    plt.close(figure)


def test_plot_spectrum_can_include_q1_as_an_opt_in():
    """q=1 is omitted by default but included when explicitly requested."""
    import matplotlib.pyplot as plt

    modes = np.arange(1, 6)
    measured = np.asarray(HSS97(np.maximum(modes, 2), kC=25.0, sigma=2.0, lmax=30))
    figure, (default_axis, q1_axis) = plt.subplots(1, 2)
    default = plot_spectrum(
        SpectrumPlotData(modes=modes, avg_amps2=measured), ax=default_axis,
        color="black",
        config=SpectrumPlotConfig(fitting_region="none"),
    )
    with_q1 = plot_spectrum(
        SpectrumPlotData(modes=modes, avg_amps2=measured), ax=q1_axis,
        color="black",
        config=SpectrumPlotConfig(fitting_region="none", include_q1=True),
    )
    assert default.nonfit_artist.get_xdata()[0] == 2
    assert with_q1.nonfit_artist.get_xdata()[0] == 1
    plt.close(figure)


from vesmod.EdgeMod.spectrum_utils import HSS97_with_camera_integration_time
from vesmod.EdgeMod.spectrum_plotting import plot_q3_scaled_spectrum


@pytest.mark.parametrize('power', ['measured', 'corrected', 'both'])
def test_camera_power_views_recover_instantaneous_spectrum(power):
    """Real camera physics supplies the synthetic powers, including nonfit modes."""
    config = SpectrumFitConfig(exposure_time=.030, viscosity_in=.00102,
                               viscosity_out=.00097, lmax=50)
    modes = np.arange(2, 12)
    instantaneous = np.asarray(HSS97(modes, 25, 2, config.lmax))
    def camera(q):
        return HSS97_with_camera_integration_time(q, 25, 2, config.lmax,
                                                  config=config, radii=[5])
    measured = np.asarray(camera(modes))
    original = measured.copy()
    data = SpectrumPlotData(modes, measured,
                            SimpleNamespace(best_values={'kC': 25, 'sigma': 2}, eval=camera),
                            3, 8, config.lmax, exposure_time=config.exposure_time)
    result = plot_spectrum(data, config=SpectrumPlotConfig(power=power))
    selected = (modes >= 3) & (modes < 8)
    if power != 'measured':
        np.testing.assert_allclose(result.corrected_fit_data_artist.get_ydata(),
                                   instantaneous[selected], rtol=1e-14)
        np.testing.assert_allclose(result.corrected_nonfit_artist.get_ydata(),
                                   instantaneous[~selected], rtol=1e-14)
        np.testing.assert_allclose(result.corrected_fit_artist.get_ydata(),
                                   instantaneous[selected], rtol=1e-14)
    if power != 'corrected':
        np.testing.assert_array_equal(result.fit_data_artist.get_ydata(), measured[selected])
        np.testing.assert_allclose(result.fit_artist.get_ydata(), measured[selected])
    np.testing.assert_array_equal(measured, original)
    plt.close(result.figure)
    figure, axis = plot_q3_scaled_spectrum(data, config=SpectrumPlotConfig(power=power))
    target = measured if power == 'measured' else instantaneous
    # Both views draw corrected points last; q^3 scaling applies exactly once.
    point_lines = [line for line in axis.lines if line.get_marker() in {'o', 's'}]
    np.testing.assert_allclose(point_lines[-1].get_ydata(),
                               modes[selected]**3 * target[selected], rtol=1e-14)
    plt.close(figure)


@pytest.mark.parametrize('power', ['corrected', 'both'])
def test_corrected_plot_requires_camera_fit(power):
    with pytest.raises(ValueError, match='camera integration fit'):
        plot_spectrum(SpectrumPlotData(np.arange(2, 5), np.ones(3)),
                      config=SpectrumPlotConfig(power=power))


def test_invalid_power_view():
    with pytest.raises(ValueError, match='power'):
        SpectrumPlotConfig(power='unknown')


def test_public_spectrum_plot_uses_recorded_camera_fit():
    config = SpectrumFitConfig(exposure_time=.030, viscosity_in=.00102,
                               viscosity_out=.00097, lmax=50, free_sigma=False)
    spectrum = Spectrum.from_radii(np.full((2, 32), 5.0))
    positive = spectrum.modes >= 2
    q = spectrum.modes[positive]
    spectrum.avg_amps2[positive] = HSS97_with_camera_integration_time(
        q, 25, 0, 50, config=config, radii=[5])
    original = spectrum.avg_amps2.copy()
    spectrum.extract_kc_from_fit(config)
    result = spectrum.plot(plot_config=SpectrumPlotConfig(power='both'))
    np.testing.assert_allclose(result.corrected_fit_data_artist.get_ydata(),
                               HSS97(np.arange(3, 8), 25, 0, 50), rtol=1e-5)
    np.testing.assert_array_equal(spectrum.avg_amps2, original)
    plt.close(result.figure)
