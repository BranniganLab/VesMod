"""Tests for the selectable temporal power used by EdgeMod."""

import numpy as np
import pytest

from vesmod.EdgeMod import Spectrum


def test_mean_square_retains_static_mode_power():
    """The current power definition includes a shape fixed across frames."""
    theta = 2 * np.pi * np.arange(120) / 120
    radii = np.array([10.0 + 0.2 * np.cos(4 * theta)] * 20)

    spectrum = Spectrum.from_radii(radii, power_definition="mean_square")

    q4 = np.flatnonzero(spectrum.modes == 4)[0]
    assert spectrum.avg_amps2[q4] > 0


def test_temporal_variance_removes_static_mode_power():
    """Subtracting the complex temporal mean removes a fixed contour mode."""
    theta = 2 * np.pi * np.arange(120) / 120
    radii = np.array([10.0 + 0.2 * np.cos(4 * theta)] * 20)

    spectrum = Spectrum.from_radii(radii, power_definition="temporal_variance")

    q4 = np.flatnonzero(spectrum.modes == 4)[0]
    assert spectrum.avg_amps2[q4] == pytest.approx(0.0, abs=1e-30)


def test_power_definitions_agree_when_complex_mode_mean_is_zero():
    """Opposite phases have the same mean-square and centered variance."""
    theta = 2 * np.pi * np.arange(120) / 120
    base = 10.0 + 0.2 * np.cos(4 * theta)
    radii = np.array([base, 10.0 - 0.2 * np.cos(4 * theta)] * 10)

    mean_square = Spectrum.from_radii(radii, power_definition="mean_square")
    temporal_variance = Spectrum.from_radii(
        radii,
        power_definition="temporal_variance",
    )

    q4 = np.flatnonzero(mean_square.modes == 4)[0]
    assert temporal_variance.avg_amps2[q4] == pytest.approx(
        mean_square.avg_amps2[q4]
    )


def test_temporal_variance_requires_two_frames():
    """A single frame cannot define temporal variance."""
    with pytest.raises(ValueError, match="at least two frames"):
        Spectrum.from_radii(
            np.full((1, 120), 10.0),
            power_definition="temporal_variance",
        )


def test_unknown_power_definition_is_rejected():
    """Only the two documented power definitions are accepted."""
    with pytest.raises(ValueError, match="power_definition must be"):
        Spectrum.from_radii(np.full((2, 120), 10.0), power_definition="variance")


def test_power_definition_is_serialized():
    """Saved spectrum metadata records which power was calculated."""
    spectrum = Spectrum.from_radii(
        np.full((2, 120), 10.0),
        power_definition="temporal_variance",
    )

    assert spectrum.to_dict(include_arrays=False)["power_definition"] == (
        "temporal_variance"
    )
