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
            {"population_bic_threshold": np.nan},
            "population_bic_threshold must be finite",
        ),
        (
            {"max_minor_population_fraction": 0.5},
            "max_minor_population_fraction must be greater than or equal to 0 and less than 0.5",
        ),
    ],
)
def test_edge_qc_config_rejects_invalid_values(kwargs, match):
    """Test representative invalid QC configuration values."""
    config_values = {
        "curvature_threshold": 5.0,
        "population_bic_threshold": 10.0,
        "max_minor_population_fraction": 0.25,
    }
    config_values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        EdgeQCConfig(**config_values)
