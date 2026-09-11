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


def test_extract_edge_from_frame_reextracts_from_refined_origin(monkeypatch):
    """A shifted contour centroid becomes the origin of a new image extraction."""
    image = np.zeros((80, 80), dtype=float)
    first_radii = np.full(80, 12.0)
    second_radii = np.full(80, 12.5)
    extraction_origins = []
    centroid_calls = []

    monkeypatch.setattr(
        edge_extractor,
        "approximate_vesicle_com",
        lambda frame: (40.0, 40.0),
    )

    def fake_extract(frame, origin):
        extraction_origins.append(origin)
        if len(extraction_origins) == 1:
            return first_radii
        return second_radii

    def fake_centroid(origin, radii):
        centroid_calls.append((origin, radii.copy()))
        if len(centroid_calls) == 1:
            return (42.0, 41.0)
        return origin

    monkeypatch.setattr(edge_extractor, "_extract_edge_from_origin", fake_extract)
    monkeypatch.setattr(edge_extractor, "radial_contour_centroid", fake_centroid)

    radii, origin = extract_edge_from_frame(image)

    assert extraction_origins == [(40.0, 40.0), (41.0, 42.0)]
    np.testing.assert_array_equal(radii, second_radii)
    assert origin == (41.0, 42.0)


def test_extract_edge_from_frame_returns_radii_measured_from_returned_origin(monkeypatch):
    """At the refinement limit, final radii are measured from the stored origin."""
    image = np.zeros((80, 80), dtype=float)
    extraction_origins = []

    monkeypatch.setattr(
        edge_extractor,
        "approximate_vesicle_com",
        lambda frame: (10.0, 20.0),
    )

    def fake_extract(frame, origin):
        extraction_origins.append(origin)
        return np.full(8, len(extraction_origins), dtype=float)

    def fake_centroid(origin, radii):
        return (origin[0] + 1.0, origin[1] + 2.0)

    monkeypatch.setattr(edge_extractor, "_extract_edge_from_origin", fake_extract)
    monkeypatch.setattr(edge_extractor, "radial_contour_centroid", fake_centroid)

    radii, origin = extract_edge_from_frame(image)

    assert extraction_origins == [
        (10.0, 20.0),
        (12.0, 21.0),
        (14.0, 22.0),
        (16.0, 23.0),
    ]
    assert origin == extraction_origins[-1]
    np.testing.assert_array_equal(radii, np.full(8, 4.0))
