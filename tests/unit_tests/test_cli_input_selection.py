"""Tests for shared multi-path and glob CLI input selection."""

from pathlib import Path
import sys

from vesmod.cli import edgemod_cli, vesedge_cli
from vesmod.cli.input_selection import select_input_files


def test_select_input_files_accepts_multiple_explicit_files(tmp_path):
    """Shell-expanded globs can arrive as multiple explicit positional files."""
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    first.touch()
    second.touch()

    paths, root = select_input_files([first, second], ".npz", recursive=False)

    assert paths == [first.resolve(), second.resolve()]
    assert root == tmp_path.resolve()


def test_select_input_files_matches_unexpanded_glob(tmp_path):
    """A quoted glob pattern is expanded by the CLI when the shell does not."""
    keep = tmp_path / "sample_a.npz"
    reject = tmp_path / "other.npz"
    keep.touch()
    reject.touch()

    paths, root = select_input_files(
        tmp_path / "sample*.npz",
        ".npz",
        recursive=False,
    )

    assert paths == [keep.resolve()]
    assert root == tmp_path.resolve()


def test_select_input_files_matches_glob_recursively(tmp_path):
    """--recursive extends a filename glob through subdirectories."""
    top = tmp_path / "sample_top.npz"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested = nested_dir / "sample_nested.npz"
    reject = nested_dir / "other.npz"
    for path in (top, nested, reject):
        path.touch()

    paths, root = select_input_files(
        tmp_path / "sample*.npz",
        ".npz",
        recursive=True,
    )

    assert paths == [nested.resolve(), top.resolve()]
    assert root == tmp_path.resolve()


def test_select_input_files_deduplicates_mixed_selectors(tmp_path):
    """Overlapping explicit and glob selectors produce each file only once."""
    path = tmp_path / "sample.npz"
    path.touch()

    paths, _ = select_input_files(
        [path, tmp_path / "sample*.npz"],
        ".npz",
        recursive=False,
    )

    assert paths == [path.resolve()]


def test_vesedge_qc_parser_accepts_multiple_inputs(monkeypatch, tmp_path):
    """VesEdge QC accepts shell-expanded multiple positional checkpoints."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "qc",
            "a.npz",
            "b.npz",
            "--output-dir",
            str(tmp_path),
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.input_path == [Path("a.npz"), Path("b.npz")]


def test_vesedge_internal_structures_parser_accepts_multiple_inputs(
    monkeypatch,
    tmp_path,
):
    """Internal-structures accepts multiple positional checkpoint selectors."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "internal-structures",
            "a.npz",
            "b.npz",
            "--output-dir",
            str(tmp_path),
            "--include-unqced",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.input_path == Path("a.npz")
    assert args.additional_input_paths == [Path("b.npz")]


def test_vesedge_gif_parser_accepts_multiple_inputs(monkeypatch, tmp_path):
    """GIF generation accepts shell-expanded multiple positional checkpoints."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "gif",
            "a.npz",
            "b.npz",
            "--output-dir",
            str(tmp_path),
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.input_path == [Path("a.npz"), Path("b.npz")]


def test_edgemod_parser_accepts_multiple_inputs(monkeypatch):
    """Core EdgeMod accepts shell-expanded multiple positional arrays."""
    monkeypatch.setattr(sys, "argv", ["edgemod", "a.npy", "b.npy"])

    args = edgemod_cli.parse_args()

    assert args.input_path == [Path("a.npy"), Path("b.npy")]
