"""Tests for in-memory and on-demand video frame sequences."""

import numpy as np
import pytest

from vesmod.VesEdge.frame_source import (
    InMemoryFrameSequence,
    OnDemandFrameSequence,
    as_frame_source,
    open_frame_source,
)
from vesmod.VesEdge import frame_source


def test_in_memory_frame_sequence_supports_indexed_and_iterative_reads():
    frames = np.arange(24).reshape(3, 4, 2)
    sequence = InMemoryFrameSequence(frames)

    assert sequence.shape == (3, 4, 2)
    np.testing.assert_array_equal(sequence[1], frames[1])
    assert len(list(sequence)) == 3


def test_as_frame_source_preserves_existing_sequence():
    sequence = InMemoryFrameSequence(np.zeros((2, 3, 4)))

    assert as_frame_source(sequence) is sequence


def test_open_numpy_source_uses_on_demand_memory_mapping(tmp_path):
    path = tmp_path / "video.npy"
    np.save(path, np.zeros((2, 3, 4)))

    with open_frame_source(path) as sequence:
        assert isinstance(sequence, OnDemandFrameSequence)
        assert sequence.shape == (2, 3, 4)
        assert isinstance(sequence._array, np.memmap)
    assert sequence._array is None


def test_in_memory_sequence_close_preserves_caller_owned_array():
    frames = np.zeros((2, 3, 4))
    sequence = InMemoryFrameSequence(frames)

    sequence.close()
    sequence.close()

    assert sequence._frames is frames


def test_in_memory_sequence_rejects_non_video_shape():
    with pytest.raises(IndexError, match="3D array"):
        InMemoryFrameSequence(np.zeros((3, 4)))


def test_on_demand_sequence_requires_explicit_nd2_multidimensional_selection(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(frame_source.nd2, "ND2File", _FakeND2File)

    with pytest.raises(ValueError, match="provide an explicit selection"):
        open_frame_source(tmp_path / "video.nd2")


def test_on_demand_sequence_reads_only_selected_nd2_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_source.nd2, "ND2File", _FakeND2File)

    with open_frame_source(
        tmp_path / "video.nd2",
        axis_selection={"Z": 1},
    ) as sequence:
        assert isinstance(sequence, OnDemandFrameSequence)
        assert sequence.shape == (2, 3, 4)
        np.testing.assert_array_equal(sequence[0], np.full((3, 4), 1))
        np.testing.assert_array_equal(sequence[1], np.full((3, 4), 3))
        assert sequence._file.read_indices == [1, 3]


def test_on_demand_nd2_sequence_recycles_reader_after_memory_budget(
    tmp_path,
    monkeypatch,
):
    _TrackingND2File.instances = []
    monkeypatch.setattr(frame_source.nd2, "ND2File", _TrackingND2File)
    monkeypatch.setattr(frame_source, "_ND2_READER_MEMORY_BUDGET_BYTES", 95)

    with open_frame_source(
        tmp_path / "video.nd2",
        axis_selection={"Z": 1},
    ) as sequence:
        first = sequence[0]
        first_reader = _TrackingND2File.instances[0]
        second = sequence[1]
        second_reader = _TrackingND2File.instances[1]

        assert first.flags.owndata
        np.testing.assert_array_equal(first, np.full((3, 4), 1))
        np.testing.assert_array_equal(second, np.full((3, 4), 3))
        assert first_reader.closed
        assert first_reader.read_indices == [1]
        assert second_reader.read_indices == [3]


def test_on_demand_nd2_sequence_returns_frames_independent_of_reader(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(frame_source.nd2, "ND2File", _FakeND2File)

    with open_frame_source(
        tmp_path / "video.nd2",
        axis_selection={"Z": 1},
    ) as sequence:
        frame = sequence[0]
        assert frame.flags.owndata

    np.testing.assert_array_equal(frame, np.full((3, 4), 1))


def test_on_demand_nd2_sequence_copies_only_selected_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_source.nd2, "ND2File", _FakeMultiChannelND2File)

    with open_frame_source(
        tmp_path / "video.nd2",
        axis_selection={"C": 1},
    ) as sequence:
        frame = sequence[0]

        np.testing.assert_array_equal(frame, np.full((3, 4), 1))
        assert frame.flags.owndata
        assert frame.base is None
        assert sequence._bytes_since_reopen == 2 * frame.nbytes


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


class _TrackingND2File(_FakeND2File):
    instances = []

    def __init__(self, path):
        super().__init__(path)
        self.instances.append(self)


class _FakeMultiChannelND2File:
    sizes = {"T": 2, "C": 2, "Y": 3, "X": 4}
    loop_indices = [{"T": 0}, {"T": 1}]

    def __init__(self, path):
        self.path = path
        self.closed = False

    def read_frame(self, index):
        return np.stack(
            [
                np.full((3, 4), index * 10),
                np.full((3, 4), index * 10 + 1),
            ]
        )

    def close(self):
        self.closed = True
