"""Tests for EdgeMod SpectrumFitConfig."""

import pytest

from vesmod.EdgeMod import (
    FixedFitRangeSelector,
    QMinusThreeFitRangeSelector,
    SpectrumFitConfig,
)


def test_default_config_preserves_historical_fit_settings():
    """Test defaults reproduce the existing fixed q=3--7 fit."""
    config = SpectrumFitConfig()

    assert config.lmax == 500
    assert config.free_sigma is True
    assert config.temperature == 295.0
    assert isinstance(config.range_selector, FixedFitRangeSelector)
    assert config.range_selector.lower_bound == 3
    assert config.range_selector.upper_bound == 8


def test_config_serializes_dynamic_selector_parameters():
    """Test scientific fit settings are retained for reproducibility."""
    config = SpectrumFitConfig(
        lmax=400,
        free_sigma=False,
        temperature=310.0,
        range_selector=QMinusThreeFitRangeSelector(
            lower_bound=3,
            upper_bound=15,
            min_modes=5,
            slope_tolerance=0.2,
            max_log_rmse=0.1,
        ),
    )

    data = config.to_dict()

    assert data["lmax"] == 400
    assert data["free_sigma"] is False
    assert data["temperature"] == 310.0
    assert data["range_selector"] == {
        "type": "QMinusThreeFitRangeSelector",
        "lower_bound": 3,
        "upper_bound": 15,
        "min_modes": 5,
        "slope_tolerance": 0.2,
        "max_log_rmse": 0.1,
    }


@pytest.mark.parametrize("lmax", [0, -1])
def test_config_rejects_nonpositive_lmax(lmax):
    """Test lmax must be positive."""
    with pytest.raises(ValueError, match="lmax must be positive"):
        SpectrumFitConfig(lmax=lmax)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_config_rejects_nonpositive_temperature(temperature):
    """Test temperature must be physically positive."""
    with pytest.raises(ValueError, match="temperature must be positive"):
        SpectrumFitConfig(temperature=temperature)
