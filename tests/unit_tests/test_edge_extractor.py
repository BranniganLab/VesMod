import pytest
import numpy as np

from vesmod.VesEdge import edge_extractor
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


def test_extract_edge_from_frame_returns_recentered_contour(monkeypatch):
    """The stored origin and radii come from contour recentering, not the detection seed."""
    image = np.zeros((80, 80), dtype=float)
    image[20:60, 20:60] = 1.0
    expected_radii = np.full(80, 12.5)
    expected_origin_xy = (31.5, 42.5)
    calls = []

    monkeypatch.setattr(
        edge_extractor,
        "approximate_vesicle_com",
        lambda frame: (40.0, 40.0),
    )

    def fake_recenter(origin, radii):
        calls.append((origin, radii.copy()))
        return expected_origin_xy, expected_radii

    monkeypatch.setattr(edge_extractor, "recenter_radial_contour", fake_recenter)

    radii, origin = extract_edge_from_frame(image)

    assert calls
    assert calls[0][0] == (40.0, 40.0)
    np.testing.assert_array_equal(radii, expected_radii)
    assert origin == (expected_origin_xy[1], expected_origin_xy[0])
