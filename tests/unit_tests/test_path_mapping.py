from pathlib import Path

import pytest

from vesmod.io import map_output_path, relative_selected_path


def test_map_output_path_preserves_nested_layout_and_replaces_suffix(tmp_path):
    root = tmp_path / "input"
    source = root / "condition" / "sample.npz"
    source.parent.mkdir(parents=True)
    source.touch()

    assert map_output_path(source, root, tmp_path / "output", suffix=".npy") == (
        tmp_path / "output" / "condition" / "sample.npy"
    )


def test_explicit_file_selector_maps_by_filename(tmp_path):
    source = tmp_path / "input" / "sample.npz"
    source.parent.mkdir()
    source.touch()

    assert relative_selected_path(source, source) == Path("sample.npz")
    assert map_output_path(source, source, tmp_path / "output", suffix=".gif") == (
        tmp_path / "output" / "sample.gif"
    )


def test_mapping_rejects_path_outside_selector_root(tmp_path):
    root = tmp_path / "input"
    source = tmp_path / "elsewhere" / "sample.npz"
    root.mkdir()
    source.parent.mkdir()
    source.touch()

    with pytest.raises(ValueError, match="outside selector root"):
        map_output_path(source, root, tmp_path / "output")
