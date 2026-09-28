"""Tests for experimental per-mode apparent bending rigidity."""

import numpy as np
import pytest

from vesmod.EdgeMod import Spectrum, SpectrumFit, SpectrumFitConfig
from vesmod.EdgeMod.experimental import (
    ModeRigidityResult,
    calculate_mode_rigidity,
    plot_mode_rigidity,
)
from vesmod.EdgeMod.spectrum_utils import (
    HSS97,
    calc_tension_from_reduced_tension,
)


def _model_case(power_definition="mean_square"):
    """Build a spectrum that exactly follows HSS97 for a known fit."""
    config = SpectrumFitConfig(
        lower_bound=3,
        upper_bound=8,
        lmax=20,
        free_sigma=True,
        temperature=300.0,
    )
    spectrum = Spectrum.from_radii(
        np.full((3, 120), 10.0),
        power_definition=power_definition,
    )
    modes = np.arange(0, 9)
    powers = np.zeros(modes.shape, dtype=float)
    positive_modes = modes[2:]
    powers[2:] = HSS97(positive_modes.tolist(), 25.0, 4.0, config.lmax)
    spectrum.modes = modes
    spectrum.avg_amps2 = powers
    fit = SpectrumFit(
        kC=25.0,
        surface_tension=calc_tension_from_reduced_tension(
            spectrum.r0,
            4.0,
            25.0,
            config.temperature,
        ),
        lower_bound=config.lower_bound,
        upper_bound=config.upper_bound,
        config=config,
        power_definition=power_definition,
    )
    return spectrum, fit


def test_model_spectrum_produces_constant_apparent_rigidity():
    """Inverting an exact HSS97 spectrum recovers its known kC at each q."""
    spectrum, fit = _model_case()

    result = calculate_mode_rigidity(spectrum, fit)

    np.testing.assert_array_equal(result.modes, np.arange(2, 9))
    np.testing.assert_allclose(result.apparent_kc, 25.0)
    assert result.reduced_sigma == pytest.approx(4.0)


def test_max_mode_is_inclusive_and_zero_power_is_nan():
    """The requested top mode is included and zero power is safely missing."""
    spectrum, fit = _model_case()
    spectrum.avg_amps2[4] = 0.0

    result = calculate_mode_rigidity(spectrum, fit, max_mode=6)

    np.testing.assert_array_equal(result.modes, np.arange(2, 7))
    assert np.isnan(result.apparent_kc[result.modes == 4]).all()
    assert np.isfinite(result.apparent_kc[result.modes != 4]).all()


def test_modes_above_lmax_are_excluded():
    """The theoretical sum is evaluated only for supported modes."""
    spectrum, fit = _model_case()
    fit = SpectrumFit(
        kC=fit.kC,
        surface_tension=fit.surface_tension,
        lower_bound=fit.lower_bound,
        upper_bound=fit.upper_bound,
        config=SpectrumFitConfig(
            lower_bound=3,
            upper_bound=7,
            lmax=6,
            free_sigma=True,
            temperature=300.0,
        ),
    )
    spectrum.avg_amps2[-1] = 1.0
    spectrum.modes = np.append(spectrum.modes, 10)
    spectrum.avg_amps2 = np.append(spectrum.avg_amps2, 1.0)

    result = calculate_mode_rigidity(spectrum, fit)

    assert result.modes.max() == 6


def test_fit_and_spectrum_power_definitions_must_match():
    """The diagnostic cannot pair a fit to a different measured power."""
    spectrum, fit = _model_case("mean_square")
    fit = SpectrumFit(
        kC=fit.kC,
        surface_tension=fit.surface_tension,
        lower_bound=fit.lower_bound,
        upper_bound=fit.upper_bound,
        config=fit.config,
        power_definition="temporal_variance",
    )

    with pytest.raises(ValueError, match="different power definitions"):
        calculate_mode_rigidity(spectrum, fit)


def test_plot_mode_rigidity_adds_data_and_fit_reference_line():
    """The plotting helper accepts an axis and draws the fitted-kC guide."""
    spectrum, fit = _model_case()
    result = calculate_mode_rigidity(spectrum, fit)

    figure, ax = plot_mode_rigidity(result, label="mean square")

    assert figure is ax.figure
    assert ax.get_xlabel() == "Fourier mode q"
    assert len(ax.lines) == 3
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), np.arange(3, 8))
    assert ax.lines[-1].get_ydata()[0] == pytest.approx(fit.kC)


def test_plot_rejects_wrong_result_type():
    """The plotting helper reports misuse clearly."""
    with pytest.raises(TypeError, match="ModeRigidityResult"):
        plot_mode_rigidity(object())
