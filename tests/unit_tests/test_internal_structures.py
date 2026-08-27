"""Tests for experimental internal-structure measurements."""

import numpy as np

from vesmod.VesEdge.internal_structures import (
    InternalStructureConfig,
    detect_internal_structures,
    summarize_internal_structures,
)
from vesmod.VesEdge.models import ImageContour


def _circular_contour(center=(60.0, 60.0), radius=50.0):
    return ImageContour(center, np.full(240, radius))


def _base_frame(seed=4):
    rng = np.random.default_rng(seed)
    return 100.0 + rng.normal(0.0, 0.3, size=(120, 120))


def _config():
    return InternalStructureConfig(
        membrane_exclusion_px=4,
        background_sigma_px=30.0,
        threshold_sigma=4.0,
        min_region_area_px=9,
        light_grow_sigma=1.25,
        min_light_circularity=0.2,
        min_light_solidity=0.8,
        max_light_eccentricity=0.95,
        structure_boundary_exclusion_px=8,
        filament_seed_threshold=0.7,
        filament_grow_threshold=0.35,
        filament_scales_px=(1.0, 2.0, 3.0, 4.0),
        min_filament_length_px=20,
        bubble_edge_sigma=2.0,
        bubble_edge_grow_sigma=1.0,
        bubble_closing_px=2,
        min_bubble_area_px=100,
        min_bubble_boundary_fraction=0.4,
        min_bubble_circularity=0.2,
        min_bubble_solidity=0.8,
        max_bubble_eccentricity=0.95,
        max_bubble_area_fraction=0.5,
    )


def test_large_light_region_grows_beyond_high_confidence_seed():
    frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    light_disk = (yy - 45) ** 2 + (xx - 45) ** 2 <= 12**2
    frame[light_disk] += 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert result.light_area_fraction > 0.0
    assert np.count_nonzero(result.light_region_mask) > 0.7 * np.count_nonzero(
        light_disk
    )


def test_amorphous_light_region_is_not_classified_as_vesicle():
    frame = _base_frame()
    frame[43:48, 27:88] += 10.0
    frame[38:65, 52:59] += 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert not np.any(result.light_region_mask)


def test_oval_light_region_is_classified_as_vesicle():
    frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    light_oval = ((yy - 45) / 8) ** 2 + ((xx - 48) / 13) ** 2 <= 1.0
    frame[light_oval] += 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert np.count_nonzero(result.light_region_mask) > 0.7 * np.count_nonzero(
        light_oval
    )
    assert any(
        region.structure_type == "light_region" for region in result.regions
    )


def test_thin_dark_filament_is_measured_by_length():
    frame = _base_frame()
    frame[73:76, 30:91] -= 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert result.filament_length_px >= 40
    assert result.filament_area_fraction > 0.0
    assert any(
        region.structure_type == "dark_filament" for region in result.regions
    )


def test_outer_membrane_is_not_classified_as_filament():
    frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    distance = np.sqrt((yy - 60) ** 2 + (xx - 60) ** 2)
    outer_membrane = np.abs(distance - 46.0) <= 1.5
    frame[outer_membrane] -= 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert result.filament_length_px == 0


def test_dark_closed_edge_fills_neutral_bubble_interior():
    frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    distance = np.sqrt((yy - 65) ** 2 + (xx - 70) ** 2)
    dark_ring = np.abs(distance - 12.0) <= 1.5
    frame[dark_ring] -= 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert result.bubble_count == 1
    full_bubble_mask = result.to_full_frame_channel_mask("bubble")
    assert full_bubble_mask[65, 70]
    assert result.bubble_area_fraction > 0.0
    assert any(region.structure_type == "bubble" for region in result.regions)


def test_weak_connected_bubble_edge_grows_from_dark_seed():
    frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    distance = np.sqrt((yy - 65) ** 2 + (xx - 70) ** 2)
    ring = np.abs(distance - 12.0) <= 1.5
    frame[ring] -= 1.0
    frame[ring & (xx >= 70)] -= 9.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert result.bubble_count == 1
    assert result.to_full_frame_channel_mask("bubble")[65, 70]


def test_whole_interior_is_not_classified_as_bubble():
    frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    distance = np.sqrt((yy - 60) ** 2 + (xx - 60) ** 2)
    oversized_ring = np.abs(distance - 38.0) <= 1.5
    frame[oversized_ring] -= 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert result.bubble_count == 0
    assert not np.any(result.bubble_region_mask)


def test_full_frame_masks_restore_crop_offset_for_each_channel():
    frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    frame[(yy - 45) ** 2 + (xx - 45) ** 2 <= 8**2] += 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())
    full_mask = result.to_full_frame_mask()

    assert full_mask.shape == frame.shape
    assert full_mask[45, 45]
    assert not full_mask[0, 0]


def test_video_summary_retains_channel_specific_population_features():
    light_frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    light_frame[(yy - 45) ** 2 + (xx - 45) ** 2 <= 10**2] += 10.0
    populated = detect_internal_structures(
        light_frame,
        _circular_contour(),
        _config(),
    )
    empty = detect_internal_structures(
        _base_frame(seed=8),
        _circular_contour(),
        _config(),
    )

    summary = summarize_internal_structures([populated, empty])

    assert summary.n_frames == 2
    assert summary.frame_prevalence == 0.5
    assert summary.upper_area_fraction > summary.median_area_fraction
    assert summary.median_light_area_fraction > 0.0
