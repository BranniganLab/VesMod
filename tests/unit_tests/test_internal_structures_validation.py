"""Validation regressions for unified internal-structure configuration."""

import numpy as np
import pytest

from vesmod.VesEdge.internal_structures import InternalStructureConfig


def test_unified_internal_structure_config_normalizes_numpy_scalars():
    """Successful construction exposes normalized Python scalar state."""
    config = InternalStructureConfig(
        membrane_exclusion_px=np.int64(4),
        background_sigma_px=np.float64(30.0),
        threshold_sigma=np.float64(4.0),
        min_region_area_px=np.int64(9),
        light_grow_sigma=np.float64(1.25),
        min_light_circularity=np.float64(0.2),
        min_light_solidity=np.float64(0.8),
        max_light_eccentricity=np.float64(0.95),
        structure_boundary_exclusion_px=np.int64(20),
        filament_seed_threshold=np.float64(0.7),
        filament_grow_threshold=np.float64(0.35),
        filament_scales_px=(np.float64(1.0), np.float64(2.0)),
        min_filament_length_px=np.int64(20),
        bubble_edge_sigma=np.float64(2.0),
        bubble_edge_grow_sigma=np.float64(1.0),
        bubble_closing_px=np.int64(2),
        min_bubble_area_px=np.int64(100),
        min_bubble_boundary_fraction=np.float64(0.45),
        min_bubble_circularity=np.float64(0.2),
        min_bubble_solidity=np.float64(0.8),
        max_bubble_eccentricity=np.float64(0.95),
        max_bubble_area_fraction=np.float64(0.5),
    )

    assert isinstance(config.membrane_exclusion_px, int)
    assert isinstance(config.background_sigma_px, float)
    assert isinstance(config.min_region_area_px, int)
    assert isinstance(config.structure_boundary_exclusion_px, int)
    assert isinstance(config.filament_seed_threshold, float)
    assert all(isinstance(scale, float) for scale in config.filament_scales_px)
    assert isinstance(config.min_bubble_boundary_fraction, float)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"membrane_exclusion_px": True}, TypeError, "must be an integer"),
        ({"min_region_area_px": 0}, ValueError, "must be positive"),
        ({"background_sigma_px": np.inf}, ValueError, "finite and positive"),
        ({"filament_scales_px": (1.0, 0.0)}, ValueError, "finite and positive"),
        ({"bubble_closing_px": -1}, ValueError, "must be non-negative"),
        (
            {"min_bubble_boundary_fraction": 1.1},
            ValueError,
            "between zero and one",
        ),
        ({"max_bubble_area_fraction": 0.0}, ValueError, "must be positive"),
    ],
)
def test_unified_internal_structure_config_rejects_invalid_values(
    kwargs,
    error,
    match,
):
    """Representative invalid settings fail at construction."""
    with pytest.raises(error, match=match):
        InternalStructureConfig(**kwargs)
