"""Tests for core EdgeMod SpectrumFitConfig."""

import numpy as np
import pytest

from vesmod.EdgeMod import SpectrumFitConfig


def test_default_config_preserves_historical_fit_settings():
    """Test defaults reproduce the existing fixed q=3--7 fit."""
    config = SpectrumFitConfig()

    assert config.lower_bound == 3
    assert config.upper_bound == 8
    assert config.lmax == 500
    assert config.free_sigma is True
    assert config.temperature == 295.0


def test_config_serializes_physical_fit_parameters():
    """Test core scientific fit settings are retained for reproducibility."""
    config = SpectrumFitConfig(
        lower_bound=5,
        upper_bound=12,
        lmax=400,
        free_sigma=False,
        temperature=310.0,
    )

    assert config.to_dict() == {
        "lower_bound": 5,
        "upper_bound": 12,
        "lmax": 400,
        "free_sigma": False,
        "temperature": 310.0,
    }


@pytest.mark.parametrize("lmax", [0, -1])
def test_config_rejects_nonpositive_lmax(lmax):
    """Test lmax must be positive."""
    with pytest.raises(ValueError, match="lmax must be positive"):
        SpectrumFitConfig(lmax=lmax)


@pytest.mark.parametrize("bounds", [(0, 8), (-1, 8)])
def test_config_rejects_nonpositive_lower_bound(bounds):
    """Test physical-fit lower q bound must be positive."""
    with pytest.raises(ValueError, match="lower_bound must be positive"):
        SpectrumFitConfig(lower_bound=bounds[0], upper_bound=bounds[1])


def test_config_rejects_nonincreasing_bounds():
    """Test the upper q bound must exceed the lower q bound."""
    with pytest.raises(ValueError, match="upper_bound must be greater"):
        SpectrumFitConfig(lower_bound=8, upper_bound=8)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_config_rejects_nonpositive_temperature(temperature):
    """Test temperature must be physically positive."""
    with pytest.raises(ValueError, match="temperature must be positive"):
        SpectrumFitConfig(temperature=temperature)


@pytest.mark.parametrize("temperature", [np.nan, np.inf, -np.inf])
def test_config_rejects_nonfinite_temperature(temperature):
    """Test temperature must be finite before fitting or serialization."""
    with pytest.raises(ValueError, match="temperature must be finite"):
        SpectrumFitConfig(temperature=temperature)
