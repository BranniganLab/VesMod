"""Parity test for extractor outputs on ND2 frames versus saved PNG frames.

This test demonstrates that running the edge extractor on a saved PNG copy of a
frame produces the same result as running it on the corresponding frame pulled
directly from the source ND2 file.

Assumptions
-----------
1. The PNG is a lossless copy of the ND2 frame used for benchmarking.
2. The extractor is deterministic.
3. The caller provides the concrete extractor function via a fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import nd2
import numpy as np
import pytest
from PIL import Image

from vesmod.VesEdge import VesicleVideo

ExtractorFunc = Callable[[np.ndarray], tuple[np.ndarray, tuple[float, float]]]


def _load_nd2_frame(nd2_path: Path, frame_index: int) -> np.ndarray:
    """Load one 2D frame from an ND2 file.

    This matches the simple frame-selection logic used so far: the leading axis
    is treated as the frame axis, and any remaining non-spatial axes are sliced
    at index 0 until a 2D image remains.
    """
    data = nd2.imread(nd2_path)

    if data.ndim < 2:
        raise ValueError(f"Unexpected ND2 shape for {nd2_path}: {data.shape}")

    if data.ndim == 2:
        if frame_index != 0:
            raise IndexError(
                f"Frame {frame_index} requested from single-frame ND2 {nd2_path}."
            )
        frame = data
    else:
        if not 0 <= frame_index < data.shape[0]:
            raise IndexError(
                f"Frame {frame_index} out of range for {nd2_path}; "
                f"valid range is 0..{data.shape[0] - 1}."
            )
        frame = data[frame_index]
        while frame.ndim > 2:
            frame = frame[0]

    return np.asarray(frame)


def _load_png_frame(png_path: Path) -> np.ndarray:
    """Load one PNG frame into a numpy array without changing pixel values."""
    image = Image.open(png_path)
    return np.asarray(image)


def _make_single_frame_video(frame_2d: np.ndarray) -> VesicleVideo:
    """Wrap one 2D frame into a 3D stack compatible with VesicleVideo."""
    return VesicleVideo(frame_2d[np.newaxis, :, :])


@pytest.fixture
def extractor_func() -> ExtractorFunc:
    """Return the concrete edge extractor under test.

    Replace the import below with your real extractor function.
    """
    from vesmod.VesEdge import extract_edge_from_frame

    return extract_edge_from_frame


@pytest.fixture
def nd2_path() -> Path:
    """Return the ND2 file that contains the benchmark source frame."""
    return Path("/home/js2746/DOPC_Cer_fluctuations/Replacement_DOPC/ND Acquisition 10_crop.nd2")


@pytest.fixture
def png_path() -> Path:
    """Return the saved PNG copy of the benchmark frame."""
    return Path("./DOPC_benchmark/test_images/ND_Acquisition_10_crop__frame_00300.png")


@pytest.fixture
def frame_index() -> int:
    """Return the ND2 frame index corresponding to the saved PNG."""
    return 300


def test_extractor_gives_same_edge_on_png_and_corresponding_nd2_frame(
    extractor_func: ExtractorFunc,
    nd2_path: Path,
    png_path: Path,
    frame_index: int,
) -> None:
    """Assert extractor parity between an ND2 frame and its saved PNG copy.

    The test first checks that the PNG pixels still match the ND2 frame exactly.
    It then runs the extractor through VesicleVideo on both inputs and compares
    the stored outputs.
    """
    nd2_frame = _load_nd2_frame(nd2_path, frame_index)
    png_frame = _load_png_frame(png_path)

    # Guard against accidental image conversion during dataset export.
    assert nd2_frame.shape == png_frame.shape

    nd2_frame_cmp = nd2_frame.astype(np.int64, copy=False)
    png_frame_cmp = png_frame.astype(np.int64, copy=False)

    np.testing.assert_array_equal(
        png_frame_cmp,
        nd2_frame_cmp,
        err_msg=(
            "The saved PNG does not numerically match the source ND2 frame. "
            "If this fails, fix the export pipeline before blaming the extractor."
        ),
    )

    nd2_video = _make_single_frame_video(nd2_frame)
    png_video = _make_single_frame_video(png_frame)

    nd2_video.extract_edges(extractor_func)
    png_video.extract_edges(extractor_func)

    # Same extractor outcome category.
    assert png_video.status[0] == nd2_video.status[0]

    # Same predicted center.
    assert png_video.vesicle_centers[0] is not None
    assert nd2_video.vesicle_centers[0] is not None
    np.testing.assert_allclose(
        png_video.vesicle_centers[0],
        nd2_video.vesicle_centers[0],
        rtol=0.0,
        atol=0.0,
    )

    # Same radial edge.
    np.testing.assert_allclose(
        png_video.r_vals[0],
        nd2_video.r_vals[0],
        rtol=0.0,
        atol=0.0,
    )

    # Same Cartesian contour.
    np.testing.assert_allclose(
        png_video.x_vals[0],
        nd2_video.x_vals[0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        png_video.y_vals[0],
        nd2_video.y_vals[0],
        rtol=0.0,
        atol=0.0,
    )
