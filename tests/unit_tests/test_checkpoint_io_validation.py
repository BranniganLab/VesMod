"""Unit tests for VesEdge checkpoint frame-index validation."""

import numpy as np
import pytest

from vesmod.VesEdge.checkpoint_io import save_checkpoint
from vesmod.VesEdge.config import EdgeExtractionConfig
from vesmod.VesEdge.models import EdgeDetection, ImageContour


def _detection(frame_index):
    """Return a minimal detection with an explicit frame index."""
    contour = ImageContour((0.0, 0.0), np.ones(4, dtype=float))
    return EdgeDetection(
        contour,
        contour,
        frame_index=frame_index,
    )


@pytest.fixture
def extraction_config():
    """Return checkpoint-compatible extraction settings."""
    return EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=4,
    )


@pytest.mark.parametrize("frame_index", [None, "0", 0.0, True, np.bool_(False)])
def test_save_checkpoint_rejects_non_integer_frame_indices(
    tmp_path,
    extraction_config,
    frame_index,
):
    """Test invalid frame-index types raise the intended ValueError."""
    detection = _detection(frame_index)

    with pytest.raises(
        ValueError,
        match="missing or inconsistent frame indices",
    ):
        save_checkpoint(
            tmp_path / "invalid.npz",
            extraction_config,
            [detection],
        )


def test_save_checkpoint_accepts_numpy_integer_frame_index(
    tmp_path,
    extraction_config,
):
    """Test NumPy integer indices are accepted before int64 coercion."""
    detection = _detection(np.int32(0))

    save_checkpoint(
        tmp_path / "valid.npz",
        extraction_config,
        [detection],
    )

    with np.load(tmp_path / "valid.npz", allow_pickle=False) as checkpoint:
        assert checkpoint["frame_indices"].dtype == np.int64
        np.testing.assert_array_equal(checkpoint["frame_indices"], [0])


def test_save_checkpoint_still_rejects_nonsequential_integer_indices(
    tmp_path,
    extraction_config,
):
    """Test valid integer types must still match source-frame ordering."""
    detection = _detection(1)

    with pytest.raises(
        ValueError,
        match="missing or inconsistent frame indices",
    ):
        save_checkpoint(
            tmp_path / "invalid.npz",
            extraction_config,
            [detection],
        )
