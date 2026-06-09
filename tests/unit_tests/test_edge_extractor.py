import pytest
import numpy as np

from vesmod.VesEdge.edge_extractor import (
    approximate_vesicle_com,
    extract_edge_from_frame,
)


def test_approximate_vesicle_com_finds_center_of_symmetric_ring():
    """Test that approximate_vesicle_com locates the center of a synthetic symmetric vesicle image."""
    n = 101

    y, x = np.indices((n, n))
    r = np.sqrt((x - 50) ** 2 + (y - 50) ** 2)

    image = np.exp(-((r - 25) ** 2) / 2)

    center = approximate_vesicle_com(image)

    assert center[0] == pytest.approx(50, abs=2)
    assert center[1] == pytest.approx(50, abs=2)


def test_approximate_vesicle_com_creates_debug_output(tmp_path):
    """Test that requesting centroid debug output produces the expected PDF file."""
    image = np.zeros((50, 50))
    image[20:30, 20:30] = 1

    approximate_vesicle_com(image, debug_path=tmp_path)

    assert tmp_path.joinpath("centroid_process_debug.pdf").is_file()


def test_extract_edge_from_frame_debug_mode_returns_none(tmp_path):
    """Test that extract_edge_from_frame skips edge extraction and returns None values when debug output is requested."""
    n = 101
    y, x = np.indices((n, n))
    r = np.sqrt((x - 50)**2 + (y - 50)**2)
    image = np.exp(-((r - 25)**2) / 2)

    r_vals, center = extract_edge_from_frame(
        image,
        debug_path=tmp_path,
    )

    assert r_vals is None
    assert center is None
