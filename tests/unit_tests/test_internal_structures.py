"""Tests for experimental internal-structure measurements."""

import numpy as np

from vesmod.VesEdge.internal_structures import (
    InternalStructureConfig,
    detect_internal_structures,
    summarize_internal_structures,
)
from vesmod.VesEdge.models import ImageContour


def _circular_contour(center=(48.0, 36.0), radius=24.0):
    return ImageContour(center, np.full(180, radius))


def _synthetic_frame():
    rng = np.random.default_rng(4)
    frame = 100.0 + rng.normal(0.0, 0.5, size=(90, 110))
    yy, xx = np.ogrid[:90, :110]
    frame[(yy - 31) ** 2 + (xx - 42) ** 2 <= 3**2] += 15.0
    frame[(yy - 44) ** 2 + (xx - 57) ** 2 <= 4**2] -= 15.0
    return frame


def test_detects_bright_and_dark_regions_in_original_coordinates():
    result = detect_internal_structures(
        _synthetic_frame(),
        _circular_contour(),
        InternalStructureConfig(
            membrane_exclusion_px=3,
            background_sigma_px=8.0,
            threshold_sigma=4.0,
            min_region_area_px=9,
        ),
    )

    assert {region.polarity for region in result.regions} == {"bright", "dark"}
    assert result.structured_area_fraction > 0.0
    assert all(region.bbox_yx[0] > 0 for region in result.regions)


def test_full_frame_mask_restores_crop_offset():
    result = detect_internal_structures(
        _synthetic_frame(),
        _circular_contour(),
        InternalStructureConfig(
            membrane_exclusion_px=3,
            background_sigma_px=8.0,
            threshold_sigma=4.0,
            min_region_area_px=9,
        ),
    )

    full_mask = result.to_full_frame_mask()

    assert full_mask.shape == (90, 110)
    assert full_mask[31, 42]
    assert full_mask[44, 57]
    assert not full_mask[0, 0]


def test_video_summary_does_not_require_region_tracking():
    config = InternalStructureConfig(
        membrane_exclusion_px=3,
        background_sigma_px=8.0,
        threshold_sigma=4.0,
        min_region_area_px=9,
    )
    populated = detect_internal_structures(
        _synthetic_frame(),
        _circular_contour(),
        config,
    )
    empty_frame = np.random.default_rng(8).normal(100.0, 0.5, (90, 110))
    empty = detect_internal_structures(empty_frame, _circular_contour(), config)

    summary = summarize_internal_structures([populated, empty])

    assert summary.n_frames == 2
    assert summary.frame_prevalence == 0.5
    assert summary.upper_area_fraction > summary.median_area_fraction
