"""Tests for experimental median-radius dust screening."""

import json

import numpy as np
import pytest

from vesmod.VesEdge import EdgeDetection, ImageContour
from vesmod.VesEdge.experimental import (
    RadiusDeviationConfig,
    screen_radius_deviations,
)


def _detection(frame_index: int, radius: float) -> EdgeDetection:
    """Return a constant-radius detection for one source frame."""
    contour = ImageContour(
        origin=(0.0, 0.0),
        r=np.full(12, radius, dtype=float),
    )
    return EdgeDetection(
        full_contour=contour,
        analysis_contour=contour,
        frame_index=frame_index,
    )


def test_screen_rejects_dust_at_start_when_vesicle_is_majority():
    """Test result does not depend on the wrong object occurring first."""
    radii = [20.0, 21.0, 19.0, 100.0, 101.0, 99.0, 100.0]
    detections = [_detection(index, radius) for index, radius in enumerate(radii)]

    result = screen_radius_deviations(
        detections,
        RadiusDeviationConfig(0.2),
    )

    assert result.reference_radius_pixels == pytest.approx(99.0)
    assert result.accepted_positions == (3, 4, 5, 6)
    assert result.rejected_count == 3
    assert [frame.frame_index for frame in result.frames if not frame.accepted] == [
        0,
        1,
        2,
    ]


def test_screen_retains_gradual_acquisition_8_sized_radius_change():
    """Test a roughly 6% physical trajectory change survives a 20% cutoff."""
    radii = np.linspace(116.0, 110.0, 20)
    detections = [_detection(index, radius) for index, radius in enumerate(radii)]

    result = screen_radius_deviations(
        detections,
        RadiusDeviationConfig(0.2),
    )

    assert result.rejected_count == 0
    assert result.accepted_count == len(detections)


def test_screen_accepts_deviation_equal_to_threshold():
    """Test only deviations strictly above the configured limit fail."""
    detections = [
        _detection(0, 8.0),
        _detection(1, 10.0),
        _detection(2, 10.0),
    ]

    result = screen_radius_deviations(
        detections,
        RadiusDeviationConfig(0.2),
    )

    assert result.frames[0].accepted


def test_config_and_result_are_json_serializable_with_numpy_scalars():
    """Test experimental diagnostics normalize NumPy scalar values."""
    config = RadiusDeviationConfig(np.float32(0.2))
    result = screen_radius_deviations([_detection(0, 10.0)], config)

    json.dumps(result.to_dict())
    assert isinstance(config.max_relative_deviation, float)


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_config_rejects_invalid_thresholds(value):
    """Test the explicit relative threshold is finite and non-negative."""
    with pytest.raises(ValueError):
        RadiusDeviationConfig(value)
