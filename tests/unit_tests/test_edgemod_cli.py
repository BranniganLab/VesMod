"""Tests for EdgeMod command-line fit configuration."""

from argparse import Namespace
from pathlib import Path

import pytest

from vesmod.EdgeMod import FixedFitRangeSelector, QMinusThreeFitRangeSelector
from vesmod.cli.edgemod_cli import build_fit_config, output_path_for


def _args(**overrides):
    """Return standard parsed CLI arguments with optional overrides."""
    values = {
        "dynamic_range": False,
        "lower_fitting_bound": 3,
        "upper_fitting_bound": 8,
        "min_modes": None,
        "slope_tolerance": None,
        "max_log_rmse": None,
        "lmax": 500,
        "fixed_sigma": False,
        "temperature": 295.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_fit_config_uses_fixed_selector_by_default():
    """Test the CLI preserves the historical fixed-range behavior."""
    config = build_fit_config(_args())

    assert isinstance(config.range_selector, FixedFitRangeSelector)
    assert config.range_selector.lower_bound == 3
    assert config.range_selector.upper_bound == 8
    assert config.lmax == 500
    assert config.free_sigma is True
    assert config.temperature == 295.0


def test_build_fit_config_constructs_dynamic_selector():
    """Test all dynamic-selection arguments are propagated into the config."""
    config = build_fit_config(
        _args(
            dynamic_range=True,
            upper_fitting_bound=15,
            min_modes=5,
            slope_tolerance=0.2,
            max_log_rmse=0.1,
            fixed_sigma=True,
        )
    )

    assert isinstance(config.range_selector, QMinusThreeFitRangeSelector)
    assert config.range_selector.lower_bound == 3
    assert config.range_selector.upper_bound == 15
    assert config.range_selector.min_modes == 5
    assert config.range_selector.slope_tolerance == 0.2
    assert config.range_selector.max_log_rmse == 0.1
    assert config.free_sigma is False


def test_build_fit_config_requires_explicit_dynamic_thresholds():
    """Test dynamic fitting cannot silently use empirical acceptance defaults."""
    with pytest.raises(ValueError, match="--slope-tolerance"):
        build_fit_config(
            _args(
                dynamic_range=True,
                min_modes=5,
                max_log_rmse=0.1,
            )
        )


def test_dynamic_output_path_does_not_overwrite_fixed_output():
    """Test fixed and dynamic CLI runs use distinct JSON filenames."""
    path = Path("sample.npy")

    assert output_path_for(path, dynamic_range=False) == Path("sample.json")
    assert output_path_for(path, dynamic_range=True) == Path("sample.dynamic.json")
