"""Unit tests for EdgeMod spectrum fit-range selection."""

import numpy as np
import pytest

from vesmod.EdgeMod import (
    FixedFitRangeSelector,
    QMinusThreeFitRangeSelector,
)


def _selector(**overrides):
    """Return a selector with explicit baseline acceptance criteria."""
    kwargs = {
        "lower_bound": 3,
        "upper_bound": 11,
        "min_modes": 5,
        "slope_tolerance": 0.1,
        "max_log_rmse": 0.05,
    }
    kwargs.update(overrides)
    return QMinusThreeFitRangeSelector(**kwargs)


def test_fixed_selector_returns_configured_range():
    """Test fixed selection preserves lower-inclusive, upper-exclusive bounds."""
    selector = FixedFitRangeSelector(3, 8)
    modes = np.arange(0, 10)
    amplitudes = np.ones(10)

    result = selector.select(modes, amplitudes)

    assert result.accepted
    assert result.lower_bound == 3
    assert result.upper_bound == 8


def test_q_minus_three_selector_prefers_longest_acceptable_range():
    """Test an exact q^-3 spectrum selects the full trusted range."""
    modes = np.arange(3, 11)
    amplitudes = 2.0 * modes.astype(float) ** -3
    selector = _selector()

    result = selector.select(modes, amplitudes)

    assert result.accepted
    assert result.lower_bound == 3
    assert result.upper_bound == 11
    assert result.slope == pytest.approx(-3.0)
    assert result.log_rmse == pytest.approx(0.0, abs=1e-12)


def test_q_minus_three_selector_can_exclude_bad_low_q_modes():
    """Test the selector finds a later contiguous q^-3 scaling regime."""
    modes = np.arange(3, 12)
    amplitudes = modes.astype(float) ** -3
    amplitudes[:2] *= np.array([8.0, 0.2])
    selector = _selector(upper_bound=12)

    result = selector.select(modes, amplitudes)

    assert result.accepted
    assert result.lower_bound == 5
    assert result.upper_bound == 12


def test_q_minus_three_selector_rejects_spectrum_without_scaling():
    """Test a q^-2 spectrum is rejected instead of being physically fit."""
    modes = np.arange(3, 11)
    amplitudes = modes.astype(float) ** -2
    selector = _selector()

    result = selector.select(modes, amplitudes)

    assert not result.accepted
    assert result.reason == "No trusted q range satisfied the q^-3 scaling criteria."
    assert result.slope is not None
    assert result.log_rmse is not None


def test_q_minus_three_selector_ignores_scaling_outside_trusted_range():
    """Test q^-3 behavior above the allowed q range cannot rescue a spectrum."""
    modes = np.arange(3, 15)
    amplitudes = modes.astype(float) ** -2
    high_q = modes >= 9
    amplitudes[high_q] = modes[high_q].astype(float) ** -3
    selector = _selector(upper_bound=9)

    result = selector.select(modes, amplitudes)

    assert not result.accepted


def test_q_minus_three_selector_rejects_too_short_scaling_region():
    """Test a short accidental q^-3 match does not satisfy min_modes."""
    modes = np.arange(3, 11)
    amplitudes = modes.astype(float) ** -2
    short_range = (modes >= 5) & (modes <= 7)
    amplitudes[short_range] = 10.0 * modes[short_range].astype(float) ** -3
    selector = _selector(
        slope_tolerance=0.01,
        max_log_rmse=0.01,
    )

    result = selector.select(modes, amplitudes)

    assert not result.accepted


def test_q_minus_three_selector_rejects_insufficient_usable_modes():
    """Test non-positive amplitudes do not count as log-space fit modes."""
    modes = np.arange(3, 8)
    amplitudes = np.array([1.0, 0.0, np.nan, 0.1, 0.05])
    selector = _selector(upper_bound=8, min_modes=4)

    result = selector.select(modes, amplitudes)

    assert not result.accepted
    assert result.lower_bound is None
    assert result.upper_bound is None
    assert "fewer than 4 usable modes" in result.reason


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"lower_bound": 0}, "lower_bound"),
        ({"upper_bound": 3}, "upper_bound"),
        ({"min_modes": 1}, "min_modes"),
        ({"slope_tolerance": -0.1}, "slope_tolerance"),
        ({"max_log_rmse": -0.1}, "max_log_rmse"),
    ],
)
def test_q_minus_three_selector_validates_configuration(overrides, message):
    """Test invalid dynamic-selection values fail clearly."""
    with pytest.raises(ValueError, match=message):
        _selector(**overrides)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"lower_bound": 3.5}, "lower_bound"),
        ({"upper_bound": 11.5}, "upper_bound"),
        ({"min_modes": 5.5}, "min_modes"),
        ({"slope_tolerance": "0.1"}, "slope_tolerance"),
        ({"max_log_rmse": "0.05"}, "max_log_rmse"),
    ],
)
def test_q_minus_three_selector_validates_configuration_types(overrides, message):
    """Test dynamic-selection configuration types are explicit."""
    with pytest.raises(TypeError, match=message):
        _selector(**overrides)
