"""Unit tests for VesEdge configuration models."""

import numpy as np
import pytest

from vesmod.VesEdge import (
    AreaQCConfig,
    CurvatureQCConfig,
    EdgeExtractionConfig,
    EdgeQCConfig,
)
from vesmod.VesEdge.experimental import InternalVesicleQCConfig


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
    ("config_type", "kwargs", "match"),
    [
        (
            CurvatureQCConfig,
            {"threshold": -1.0},
            "threshold must be non-negative",
        ),
        (
            AreaQCConfig,
            {"max_relative_deviation": -0.1},
            "max_relative_deviation must be at least 0",
        ),
        (
            AreaQCConfig,
            {"max_relative_deviation": 1.0},
            "max_relative_deviation must be at least 0",
        ),
        (
            AreaQCConfig,
            {"max_relative_deviation": np.inf},
            "max_relative_deviation must be finite",
        ),
    ],
)
def test_edge_qc_config_rejects_invalid_values(config_type, kwargs, match):
    """Test representative invalid QC configuration values."""
    with pytest.raises(ValueError, match=match):
        config_type(**kwargs)


def test_edge_qc_config_normalizes_numeric_thresholds():
    """Successful construction exposes normalized finite threshold values."""
    config = EdgeQCConfig(
        curvature=CurvatureQCConfig(threshold=np.float64(5.0)),
        area=AreaQCConfig(max_relative_deviation=np.float64(0.25)),
    )

    assert isinstance(config.curvature.threshold, float)
    assert config.curvature.threshold == 5.0
    assert isinstance(config.area.max_relative_deviation, float)
    assert config.area.max_relative_deviation == 0.25


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CurvatureQCConfig(5.0, enabled=1),
        lambda: AreaQCConfig(enabled=1),
        lambda: InternalVesicleQCConfig(enabled=1),
    ],
)
def test_edge_qc_config_requires_boolean_enable_flags(factory):
    """QC enable flags are part of the validated config invariant."""
    with pytest.raises(TypeError, match="enabled must be a bool"):
        factory()


def test_edge_qc_config_migrates_legacy_flat_dictionary():
    """Old provenance is translated at the deserialization boundary."""
    config = EdgeQCConfig.from_dict(
        {
            "curvature_threshold": 7.0,
            "enable_area_qc": False,
        }
    )

    assert config.curvature.threshold == 7.0
    assert not config.area.enabled


def test_edge_qc_config_contains_independent_check_configs():
    """Each QC family is represented by its own immutable configuration."""
    curvature = CurvatureQCConfig(5.0, enabled=False)
    area = AreaQCConfig(0.4)
    internal = InternalVesicleQCConfig(enabled=True, max_frames=4)

    config = EdgeQCConfig(curvature, area, internal)

    assert config.curvature is curvature
    assert config.area is area
    assert config.internal_vesicle is internal


@pytest.mark.parametrize(
    "values",
    [
        {"curvature": {"threshold": 5.0}, "unknown": True},
        {"curvature": {"threshold": 5.0}, "enable_area_qc": False},
        {"curvature": {"threshold": 5.0, "unknown": True}},
    ],
)
def test_edge_qc_config_rejects_unknown_or_mixed_nested_fields(values):
    """Nested provenance cannot silently discard unsupported fields."""
    with pytest.raises(TypeError):
        EdgeQCConfig.from_dict(values)


def test_edge_qc_config_rejects_legacy_pixel_separation():
    """A fixed pixel distance cannot be silently mapped to a relative one."""
    with pytest.raises(ValueError, match="cannot be converted"):
        EdgeQCConfig.from_dict(
            {
                "curvature_threshold": 5.0,
                "internal_vesicle_min_separation_pixels": 5.0,
            }
        )
