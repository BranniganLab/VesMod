"""Tests for experimental internal-structure measurements."""

import numpy as np
import pytest
from skimage.measure import label

from vesmod.VesEdge.experimental import internal_structures
from vesmod.VesEdge.experimental.internal_structures import (
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


def test_internal_structure_config_normalizes_numpy_scalars():
    """Test shared validation normalizes scalar configuration values."""
    config = InternalStructureConfig(
        membrane_exclusion_px=np.int64(4),
        background_sigma_px=np.float64(30.0),
        threshold_sigma=np.float64(4.0),
        min_region_area_px=np.int64(9),
        light_grow_sigma=np.float64(1.25),
        min_light_circularity=np.float64(0.2),
        min_light_solidity=np.float64(0.8),
        max_light_eccentricity=np.float64(0.95),
        structure_boundary_exclusion_px=np.int64(8),
        filament_seed_threshold=np.float64(0.7),
        filament_grow_threshold=np.float64(0.35),
        filament_scales_px=(np.float64(1.0), np.float64(2.0)),
        min_filament_length_px=np.int64(8),
        bubble_edge_sigma=np.float64(2.0),
        bubble_edge_grow_sigma=np.float64(1.0),
        bubble_closing_px=np.int64(2),
        min_bubble_area_px=np.int64(25),
        min_bubble_boundary_fraction=np.float64(0.4),
        min_bubble_circularity=np.float64(0.2),
        min_bubble_solidity=np.float64(0.8),
        max_bubble_eccentricity=np.float64(0.95),
        max_bubble_area_fraction=np.float64(0.5),
    )

    assert isinstance(config.membrane_exclusion_px, int)
    assert type(config.background_sigma_px) is float
    assert isinstance(config.min_region_area_px, int)
    assert all(type(scale) is float for scale in config.filament_scales_px)
    assert type(config.min_bubble_boundary_fraction) is float


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
    ],
)
def test_internal_structure_config_rejects_invalid_values(kwargs, error, match):
    """Test representative invalid settings fail at config construction."""
    with pytest.raises(error, match=match):
        InternalStructureConfig(**kwargs)


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


def test_strong_interior_seed_can_fill_into_boundary_exclusion_band():
    shape = (100, 100)
    yy, xx = np.ogrid[:100, :100]
    seed_mask = (yy - 50) ** 2 + (xx - 50) ** 2 <= 25**2
    growth_mask = (yy - 50) ** 2 + (xx - 50) ** 2 <= 42**2
    structure = (yy - 50) ** 2 + (xx - 70) ** 2 <= 12**2
    normalized = np.zeros(shape, dtype=float)
    normalized[structure] = 5.0

    detected = internal_structures._detect_compact_regions(
        normalized,
        seed_mask,
        growth_mask,
        seed_sigma=4.0,
        grow_sigma=1.5,
        polarity=1,
        min_area_px=9,
        min_circularity=0.2,
        min_solidity=0.8,
        max_eccentricity=0.95,
    )

    assert detected[50, 81]
    assert not seed_mask[50, 81]


def test_bright_region_subtraction_halo_is_not_added_to_union():
    frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    distance = np.sqrt((yy - 45) ** 2 + (xx - 45) ** 2)
    frame[distance <= 10.0] += 20.0

    result = detect_internal_structures(frame, _circular_contour(), _config())
    full_mask = result.to_full_frame_mask()

    assert full_mask[45, 45]
    assert not np.any(full_mask[(distance >= 15.0) & (distance <= 25.0)])
    assert all(
        region.evidence_types == ("bright_region",)
        for region in result.regions
    )


def test_amorphous_light_region_is_not_supported_by_bright_compact_evidence():
    frame = _base_frame()
    frame[43:48, 27:88] += 10.0
    frame[38:65, 52:59] += 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert not np.any(result.light_region_mask)


def test_oval_light_region_has_bright_compact_evidence():
    frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    light_oval = ((yy - 45) / 8) ** 2 + ((xx - 48) / 13) ** 2 <= 1.0
    frame[light_oval] += 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert np.count_nonzero(result.light_region_mask) > 0.7 * np.count_nonzero(
        light_oval
    )
    assert any("bright_region" in region.evidence_types for region in result.regions)


def test_thin_dark_filament_is_measured_by_length():
    frame = _base_frame()
    frame[73:76, 30:91] -= 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    assert result.filament_length_px >= 40
    assert result.filament_area_fraction > 0.0
    assert any("curvilinear" in region.evidence_types for region in result.regions)


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
    assert any(
        "enclosed_boundary" in region.evidence_types
        for region in result.regions
    )


def test_filled_dark_region_does_not_require_a_neutral_hole():
    frame = _base_frame()
    yy, xx = np.ogrid[:120, :120]
    dark_disk = (yy - 65) ** 2 + (xx - 70) ** 2 <= 12**2
    frame[dark_disk] -= 10.0

    result = detect_internal_structures(frame, _circular_contour(), _config())

    full_dark_mask = result.to_full_frame_channel_mask("dark_region")
    assert full_dark_mask[65, 70]
    assert result.to_full_frame_mask()[65, 70]
    assert any("dark_region" in region.evidence_types for region in result.regions)


def test_light_bordered_dark_band_is_merged_as_one_structure():
    frame = _base_frame()
    frame[66:69, 28:92] += 7.0
    frame[69:76, 28:92] -= 8.0
    frame[76:79, 28:92] += 7.0

    result = detect_internal_structures(frame, _circular_contour(), _config())
    full_mask = result.to_full_frame_mask()

    assert np.mean(full_mask[69:76, 35:85]) > 0.7
    assert any("curvilinear" in region.evidence_types for region in result.regions)


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


def test_retained_curvilinear_ring_can_define_bubble_boundary():
    """Test a closed retained ridge contributes enclosed-boundary evidence."""
    shape = (120, 120)
    yy, xx = np.ogrid[:120, :120]
    distance = np.sqrt((yy - 60) ** 2 + (xx - 60) ** 2)
    retained_ridge = np.abs(distance - 12.0) <= 1.0
    detection_mask = (yy - 60) ** 2 + (xx - 60) ** 2 <= 40**2
    normalized = np.zeros(shape, dtype=float)
    ridge_response = np.zeros(shape, dtype=float)

    bubble_mask = internal_structures._detect_bubbles(
        normalized,
        ridge_response,
        retained_ridge,
        detection_mask,
        detection_mask,
        _config(),
    )

    assert bubble_mask[60, 60]
    assert label(bubble_mask, connectivity=2).max() == 1


def test_enclosed_boundary_suppresses_neighboring_structure_halos():
    shape = (120, 120)
    yy, xx = np.ogrid[:120, :120]
    distance = np.sqrt((yy - 55) ** 2 + (xx - 55) ** 2)
    bubble = distance <= 10.0
    halo = (distance > 10.0) & (distance <= 14.0)
    distant_filament = np.zeros(shape, dtype=bool)
    distant_filament[90:93, 35:85] = True

    bright, dark, ridge, skeleton = (
        internal_structures._suppress_enclosed_boundary_halos(
            bubble,
            halo.copy(),
            halo.copy(),
            halo | distant_filament,
            _config(),
        )
    )

    assert not np.any(bright[halo])
    assert not np.any(dark[halo])
    assert not np.any(ridge[halo])
    assert np.any(ridge[distant_filament])
    assert np.any(skeleton[distant_filament])


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
    assert summary.median_dark_region_area_fraction == 0.0
