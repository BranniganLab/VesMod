"""Unit tests for VesEdge configuration models."""

import numpy as np
import pytest

from vesmod.VesEdge import EdgeExtractionConfig, EdgeQCConfig


def test_edge_extraction_config_requires_positive_pixels_per_micron():
    """Test that pixels_per_micron must be positive."""
    with pytest.raises(ValueError):
        EdgeExtractionConfig(
            pixels_per_micron=0,
            n_angular_samples=120,
        )


def test_edge_extraction_config_accepts_numpy_real_scalars():
    """Test NumPy real scalars are accepted and normalized."""
    config = EdgeExtractionConfig(
        pixels_per_micron=np.float64(2.5),
        n_angular_samples=np.int64(120),
    )

    assert config.pixels_per_micron == 2.5
    assert isinstance(config.pixels_per_micron, float)
    assert config.n_angular_samples == 120
    assert isinstance(config.n_angular_samples, int)


def test_edge_extraction_config_rejects_bool_numeric_inputs():
    """Test booleans are not accepted as numeric extraction settings."""
    with pytest.raises(TypeError, match="pixels_per_micron must be a real number"):
        EdgeExtractionConfig(
            pixels_per_micron=True,
            n_angular_samples=120,
        )

    with pytest.raises(TypeError, match="n_angular_samples must be"):
        EdgeExtractionConfig(
            pixels_per_micron=1.0,
            n_angular_samples=True,
        )


def test_edge_extraction_config_converts_integer_valued_sample_count():
    """Test that integer-valued sample counts are normalized to int."""
    config = EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=120.0,
    )
    assert config.n_angular_samples == 120
    assert isinstance(config.n_angular_samples, int)


def test_edge_extraction_config_rejects_non_integer_sample_count():
    """Test that non-integer sample counts are rejected."""
    with pytest.raises(ValueError, match="integer-valued"):
        EdgeExtractionConfig(
            pixels_per_micron=1.0,
            n_angular_samples=120.5,
        )


def test_edge_extraction_config_allows_no_downsampling():
    """Test that None disables contour downsampling."""
    config = EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=None,
    )
    assert config.n_angular_samples is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"curvature_threshold": -1.0},
            "curvature_threshold must be non-negative",
        ),
        (
            {"max_relative_area_deviation": -0.1},
            "max_relative_area_deviation must be at least 0",
        ),
        (
            {"max_relative_area_deviation": 1.0},
            "max_relative_area_deviation must be at least 0",
        ),
        (
            {"max_relative_area_deviation": np.inf},
            "max_relative_area_deviation must be finite",
        ),
    ],
)
def test_edge_qc_config_rejects_invalid_values(kwargs, match):
    """Test representative invalid QC configuration values."""
    config_values = {
        "curvature_threshold": 5.0,
    }
    config_values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        EdgeQCConfig(**config_values)


def test_edge_qc_config_normalizes_numeric_thresholds():
    """Successful construction exposes normalized finite threshold values."""
    config = EdgeQCConfig(
        curvature_threshold=np.float64(5.0),
        max_relative_area_deviation=np.float64(0.25),
    )

    assert isinstance(config.curvature_threshold, float)
    assert config.curvature_threshold == 5.0
    assert isinstance(config.max_relative_area_deviation, float)
    assert config.max_relative_area_deviation == 0.25


@pytest.mark.parametrize("field", ["enable_curvature_qc", "enable_area_qc"])
def test_edge_qc_config_requires_boolean_enable_flags(field):
    """QC enable flags are part of the validated config invariant."""
    kwargs = {
        "curvature_threshold": 5.0,
        field: 1,
    }

    with pytest.raises(TypeError, match=f"{field} must be a bool"):
        EdgeQCConfig(**kwargs)
