"""Tests for experimental absolute temporal-RMS screening."""

import numpy as np
import pytest

from vesmod.EdgeMod.experimental import (
    TemporalRMSConfig,
    calculate_temporal_rms,
)


def test_temporal_rms_recovers_injected_mode():
    """Test combined RMS retains physical units for a moving Fourier mode."""
    angles = 2.0 * np.pi * np.arange(120) / 120
    phases = 2.0 * np.pi * np.arange(20) / 20
    amplitude_microns = 0.1
    radii = np.array(
        [10.0 + amplitude_microns * np.cos(3.0 * angles + phase) for phase in phases]
    )

    result = calculate_temporal_rms(radii)

    assert result.amplitude_nm == pytest.approx(
        1000.0 * amplitude_microns / np.sqrt(2.0)
    )
    assert result.included


def test_temporal_rms_ignores_static_non_circularity():
    """Test temporal centering removes a persistent Fourier mode."""
    angles = 2.0 * np.pi * np.arange(120) / 120
    radii = np.tile(10.0 + 0.2 * np.cos(3.0 * angles), (10, 1))

    result = calculate_temporal_rms(
        radii,
        TemporalRMSConfig(cutoff_nm=50.0),
    )

    assert result.amplitude_nm == pytest.approx(0.0, abs=1e-11)
    assert result.included is False


@pytest.mark.parametrize("cutoff", [-1.0, np.inf, np.nan])
def test_temporal_rms_config_rejects_invalid_cutoff(cutoff):
    """Test cutoff values must be finite and non-negative."""
    with pytest.raises(ValueError):
        TemporalRMSConfig(cutoff_nm=cutoff)
