from pathlib import Path

import numpy as np
import pytest

from vesmod.VesEdge.vesicle_video import VesicleVideo


def test_post_init_allocates_internal_arrays():
    """Test that construction creates storage arrays with expected dimensions and default values."""
    frames = np.zeros((3, 200, 200))

    video = VesicleVideo(frames, micron_to_pixel_ratio=0.5)

    assert video.r_vals.shape == (3, 120)
    assert video.status.shape == (3,)
    assert len(video.vesicle_centers) == 3


def test_post_init_requires_numpy_array():
    """Test that frames must be a numpy ndarray."""
    with pytest.raises(TypeError):
        VesicleVideo([], 1.0)


def test_post_init_requires_3d_array():
    """Test that frames must be three-dimensional."""
    with pytest.raises(IndexError):
        VesicleVideo(np.zeros((10, 10)), 1.0)


def test_post_init_requires_positive_micron_to_pixel_ratio():
    """Test that micron_to_pixel_ratio must be positive."""
    with pytest.raises(ValueError):
        VesicleVideo(np.zeros((2, 10, 10)), 0)


def test_post_init_requires_positive_n_angular_samples():
    """Test that n_angular_samples must be positive."""
    with pytest.raises(ValueError):
        VesicleVideo(np.zeros((2, 10, 10)), 1.0, n_angular_samples=0)


def test_post_init_rejects_too_many_samples():
    """Test that n_angular_samples cannot exceed the native contour length."""
    with pytest.raises(IndexError):
        VesicleVideo(np.zeros((2, 50, 50)), 1.0, n_angular_samples=1000)


def test_downsample_r_vals_returns_input_when_same_length():
    """Test that no interpolation occurs when the requested sample count equals the input length."""
    video = VesicleVideo(np.zeros((1, 10, 10)), 1.0, n_angular_samples=None)

    r_vals = np.arange(20)

    returned = video._downsample_r_vals(r_vals, 20)

    assert returned is r_vals


def test_add_edge_marks_good_frame():
    """Test that a smooth contour with low curvature receives status code 1."""
    video = VesicleVideo(
        np.zeros((1, 120, 120)),
        1.0,
        n_angular_samples=120,
    )

    r_vals = np.full(120, 20.0)

    video._add_edge_to_video_frame(
        0,
        r_vals,
        (50, 50),
        curvature_threshold=5,
    )

    assert video.status[0] == 1


def test_add_edge_marks_bad_curvature():
    """Test that excessive wrapped second differences result in status code 3."""
    video = VesicleVideo(
        np.zeros((1, 120, 120)),
        1.0,
        n_angular_samples=120,
    )

    r_vals = np.zeros(120)
    r_vals[0] = 100

    video._add_edge_to_video_frame(
        0,
        r_vals,
        (50, 50),
        curvature_threshold=5,
    )

    assert video.status[0] == 3


def test_extract_edges_marks_exceptions_as_status_2():
    """Test that extractor exceptions are caught and stored as status code 2."""
    video = VesicleVideo(np.zeros((2, 200, 200)), 1.0)

    def failing_extractor(frame):
        raise RuntimeError("failure")

    video.extract_edges(failing_extractor)

    np.testing.assert_array_equal(video.status, [2, 2])


def test_save_edge_to_npy_only_saves_good_frames(tmp_path):
    """Test that save_edge_to_npy writes only frames whose status code equals 1."""
    video = VesicleVideo(np.zeros((3, 200, 200)), 1.0)

    video.r_vals[:] = 1
    video.status[:] = [1, 2, 1]

    outfile = tmp_path / "edges"

    video.save_edge_to_npy(outfile)

    saved = np.load(outfile.with_suffix(".npy"))

    assert saved.shape[0] == 2


def test_save_edge_to_npy_requires_detected_edges(tmp_path):
    """Test that save_edge_to_npy raises when all edge values remain NaN."""
    video = VesicleVideo(np.zeros((2, 200, 200)), 1.0)

    with pytest.raises(AttributeError):
        video.save_edge_to_npy(tmp_path / "edges")
