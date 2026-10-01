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


@pytest.mark.parametrize("changed_row", [0, 16, 31])
def test_radial_gradient_does_not_mix_angular_rows(changed_row):
    """Changing a radial profile leaves all other angular rows unchanged."""
    rng = np.random.default_rng(42)
    polar_image = rng.normal(size=(32, 80))
    original_gradient = edge_extractor._radial_gradient(polar_image)
    polar_image[changed_row] += rng.normal(scale=10, size=80)
    changed_gradient = edge_extractor._radial_gradient(polar_image)

    unchanged_rows = np.arange(32) != changed_row
    np.testing.assert_array_equal(
        changed_gradient[unchanged_rows], original_gradient[unchanged_rows]
    )
    assert not np.array_equal(
        changed_gradient[changed_row], original_gradient[changed_row]
    )


def test_radial_gradient_preserves_signed_edges_in_integer_images():
    """An outward intensity decrease remains negative rather than wrapping."""
    polar_image = np.zeros((8, 80), dtype=np.uint16)
    polar_image[:, :40] = 1000

    gradient = edge_extractor._radial_gradient(polar_image)

    assert gradient.dtype.kind == "f"
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient <= 0)
    assert np.all(gradient[:, 39:41] < 0)


@pytest.mark.parametrize("mode", [2, 7, 12, 20])
def test_final_extraction_recovers_known_contour_mode(monkeypatch, mode):
    """Final measurement retains imposed modes inside a mode-7 search window."""
    n_angles = 360
    theta = 2 * np.pi * np.arange(n_angles) / n_angles
    amplitude = 3.0
    expected_radii = 80 + amplitude * np.cos(mode * theta)
    radial_samples = np.arange(160, dtype=float)
    polar_image = np.tanh(
        (radial_samples[None, :] - expected_radii[:, None]) / 2
    )
    # Keep localization fixed to isolate final measurement, including masking.
    localization_image = np.zeros_like(polar_image)
    localization_image[:, 80] = 1
    polar_images = iter((localization_image, polar_image))
    monkeypatch.setattr(
        edge_extractor,
        "wrap_image_to_polar",
        lambda image, origin: (next(polar_images), 1.0),
    )

    radii = edge_extractor._extract_edge_from_origin(
        np.zeros((n_angles, 160)), (180, 80)
    )

    np.testing.assert_allclose(radii, expected_radii, atol=0.5)
    recovered_amplitude = 2 * abs(np.fft.rfft(radii)[mode]) / n_angles
    assert recovered_amplitude == pytest.approx(amplitude, rel=0.05)
