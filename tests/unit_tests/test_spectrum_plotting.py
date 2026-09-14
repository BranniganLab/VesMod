"""Unit tests for EdgeMod spectrum plotting."""

from types import SimpleNamespace

import numpy as np

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
