"""Tests for schema-agnostic checkpoint source access."""

from pathlib import Path

import pytest

from vesmod.io import build_video_filename_index, resolve_source_path


def test_resolve_source_path_uses_recorded_absolute_path(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "sample.npz"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    source = tmp_path / "raw" / "video.nd2"
    source.parent.mkdir()
    source.touch()

    assert resolve_source_path(source, checkpoint) == source.resolve()


def test_resolve_source_path_resolves_recorded_relative_path_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "sample.npz"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    source = checkpoint.parent / "raw" / "video.nd2"
    source.parent.mkdir()
    source.touch()

    assert resolve_source_path(Path("raw/video.nd2"), checkpoint) == source.resolve()


def test_resolve_source_path_falls_back_to_video_root_index(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "sample.npz"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    source = tmp_path / "videos" / "nested" / "sample.nd2"
    source.parent.mkdir(parents=True)
    source.touch()
    index = build_video_filename_index([checkpoint], tmp_path / "videos")

    assert resolve_source_path(None, checkpoint, tmp_path / "videos", index) == source.resolve()


def test_resolve_source_path_rejects_ambiguous_fallback(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "sample.npz"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    root = tmp_path / "videos"
    for name in ("a", "b"):
        path = root / name / "sample.nd2"
        path.parent.mkdir(parents=True)
        path.touch()

    with pytest.raises(ValueError, match="Multiple source videos match"):
        resolve_source_path(None, checkpoint, root)


def test_resolve_source_path_reports_missing_recorded_source(tmp_path):
    checkpoint = tmp_path / "sample.npz"
    checkpoint.touch()

    with pytest.raises(FileNotFoundError, match="Source video does not exist"):
        resolve_source_path(Path("missing.nd2"), checkpoint)
