"""Regression tests for compact-region filtering after halo suppression."""

import numpy as np

from vesmod.VesEdge.internal_structures import (
    InternalStructureConfig,
    _suppress_bright_region_halos,
    _suppress_enclosed_boundary_halos,
)


def _config():
    """Return permissive shape gates with meaningful minimum areas."""
    return InternalStructureConfig(
        min_region_area_px=9,
        min_light_circularity=0.0,
        min_light_solidity=0.0,
        max_light_eccentricity=1.0,
        filament_scales_px=(1.0,),
        min_filament_length_px=1,
        min_bubble_area_px=9,
        min_bubble_circularity=0.0,
        min_bubble_solidity=0.0,
        max_bubble_eccentricity=1.0,
    )


def test_bright_halo_suppression_drops_small_dark_fragment():
    """Test masking a dark compact region cannot leave a sub-threshold shard."""
    bright = np.zeros((30, 30), dtype=bool)
    dark = np.zeros_like(bright)
    ridge = np.zeros_like(bright)
    enclosed = np.zeros_like(bright)
    bright[10:15, 10:15] = True
    dark[10:15, 17:22] = True

    dark_filtered, _, _, _ = _suppress_bright_region_halos(
        bright,
        dark,
        ridge,
        enclosed,
        _config(),
    )

    assert not dark_filtered.any()


def test_enclosed_halo_suppression_revalidates_bright_and_dark_fragments():
    """Test bubble masking reapplies compact filters to both signed channels."""
    enclosed = np.zeros((30, 30), dtype=bool)
    bright = np.zeros_like(enclosed)
    dark = np.zeros_like(enclosed)
    ridge = np.zeros_like(enclosed)
    enclosed[12:16, 12:16] = True
    bright[10:15, 8:13] = True
    dark[13:18, 17:22] = True

    bright_filtered, dark_filtered, _, _ = _suppress_enclosed_boundary_halos(
        enclosed,
        bright,
        dark,
        ridge,
        _config(),
    )

    assert not bright_filtered.any()
    assert not dark_filtered.any()
