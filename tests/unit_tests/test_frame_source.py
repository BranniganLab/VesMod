"""Tests for reusable lazy video frame sources."""

import numpy as np
import pytest

from vesmod.VesEdge.frame_source import (
    ArrayFrameSource,
    as_frame_source,
    open_frame_source,
)
from vesmod.VesEdge import frame_source


def test_array_frame_source_supports_indexed_and_iterative_reads():
    frames = np.arange(24).reshape(3, 4, 2)
    source = ArrayFrameSource(frames)

    assert source.shape == (3, 4, 2)
    np.testing.assert_array_equal(source[1], frames[1])
    assert len(list(source)) == 3


def test_as_frame_source_preserves_existing_source():
    source = ArrayFrameSource(np.zeros((2, 3, 4)))

    assert as_frame_source(source) is source


def test_open_numpy_source_uses_memory_mapping(tmp_path):
    path = tmp_path / "video.npy"
    np.save(path, np.zeros((2, 3, 4)))

    with open_frame_source(path) as source:
        assert source.shape == (2, 3, 4)
        assert isinstance(source._frames, np.memmap)
    assert source._frames is None


def test_array_source_close_preserves_caller_owned_array():
    frames = np.zeros((2, 3, 4))
    source = ArrayFrameSource(frames)

    source.close()
    source.close()

    assert source._frames is frames


def test_array_source_rejects_non_video_shape():
    with pytest.raises(IndexError, match="3D array"):
        ArrayFrameSource(np.zeros((3, 4)))


def test_nd2_source_requires_explicit_multidimensional_selection(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(frame_source.nd2, "ND2File", _FakeND2File)

    with pytest.raises(ValueError, match="provide an explicit selection"):
        open_frame_source(tmp_path / "video.nd2")


def test_nd2_source_reads_only_selected_sequence_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_source.nd2, "ND2File", _FakeND2File)

    with open_frame_source(
        tmp_path / "video.nd2",
        axis_selection={"Z": 1},
    ) as source:
        assert source.shape == (2, 3, 4)
        np.testing.assert_array_equal(source[0], np.full((3, 4), 1))
        np.testing.assert_array_equal(source[1], np.full((3, 4), 3))
        assert source._file.read_indices == [1, 3]


class _FakeND2File:
    sizes = {"T": 2, "Z": 2, "Y": 3, "X": 4}
    loop_indices = [
        {"T": 0, "Z": 0},
        {"T": 0, "Z": 1},
        {"T": 1, "Z": 0},
        {"T": 1, "Z": 1},
    ]

    def __init__(self, path):
        self.path = path
        self.read_indices = []
        self.closed = False

    def read_frame(self, index):
        self.read_indices.append(index)
        return np.full((3, 4), index)

    def close(self):
        self.closed = True
