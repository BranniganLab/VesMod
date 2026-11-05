"""Unit tests for the VesEdge command-line interface."""

import argparse
from contextlib import nullcontext
import sys
from pathlib import Path

import numpy as np
import pytest

from vesmod.cli import vesedge_cli


def _extract_args(tmp_path=None, input_path=Path("sample.nd2")) -> argparse.Namespace:
    """Return standard extraction CLI arguments."""
    return argparse.Namespace(
        input_path=input_path,
        output_dir=tmp_path,
        no_gif=True,
        overwrite=True,
        extractor_file=None,
        extractor="vesmod.VesEdge.extraction.edge_extractor:extract_edge_from_frame",
        extractor_name="extract_edge_from_frame",
        downsample=False,
        n_samples=120,
        pixels_per_micron=1.0,
    )


def test_parse_args_selects_extract_subcommand(monkeypatch):
    """Test extraction arguments are scoped to the extract subcommand."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "extract",
            "sample.nd2",
            "--pixels-per-micron",
            "13.44",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.command == "extract"
    assert args.input_path == Path("sample.nd2")
    assert args.pixels_per_micron == pytest.approx(13.44)


def test_iter_input_files_accepts_case_insensitive_suffixes(tmp_path):
    """Test directory discovery treats suffix case like direct-file validation."""
    lower = tmp_path / "lower.npz"
    upper = tmp_path / "upper.NPZ"
    ignored = tmp_path / "other.txt"
    lower.touch()
    upper.touch()
    ignored.touch()

    paths = vesedge_cli.iter_input_files(tmp_path, ".npz", recursive=False)

    assert paths == sorted([lower, upper])


def test_output_base_preserves_relative_subdirectories(tmp_path):
    """Test equal stems in different input folders receive distinct outputs."""
    input_root = tmp_path / "inputs"
    first = input_root / "a" / "sample.nd2"
    second = input_root / "b" / "sample.nd2"
    output_dir = tmp_path / "outputs"

    first_output = vesedge_cli._output_base(first, input_root, output_dir)
    second_output = vesedge_cli._output_base(second, input_root, output_dir)

    assert first_output == output_dir / "a" / "sample"
    assert second_output == output_dir / "b" / "sample"


def test_process_extract_file_reports_failure_and_returns(monkeypatch, capsys):
    """Test that one extraction failure does not abort batch processing."""

    class FailingVideo:
        def __init__(self, frames):
            pass

        def extract_edges(self, extractor, extraction_config):
            raise ValueError("no successful detections")

    path = Path("failed.nd2")
    monkeypatch.setattr(
        vesedge_cli,
        "open_frame_source",
        lambda input_path: nullcontext(np.zeros((1, 10, 10))),
    )
    monkeypatch.setattr(
        vesedge_cli,
        "load_extractor_from_module",
        lambda import_string: object(),
    )
    monkeypatch.setattr(vesedge_cli, "VesicleVideo", FailingVideo)

    vesedge_cli.process_extract_file(path, _extract_args(input_path=path))

    captured = capsys.readouterr()
    assert (
        f"Failed to extract {path.resolve()}: no successful detections"
        in captured.out
    )


def test_process_extract_file_saves_checkpoint_without_running_qc(
    tmp_path,
    monkeypatch,
):
    """Test extraction writes a reusable checkpoint and does not run QC."""
    observed = {}

    class FakeEdges:
        def save_checkpoint(self, path):
            observed["checkpoint"] = path

        def run_qc(self, qc_config):
            raise AssertionError("Extraction should not run QC")

    class SuccessfulVideo:
        def __init__(self, frames):
            observed["frames"] = frames

        def extract_edges(self, extractor, extraction_config):
            observed["extraction_config"] = extraction_config
            return FakeEdges()

    path = Path("sample.nd2")
    monkeypatch.setattr(
        vesedge_cli,
        "open_frame_source",
        lambda input_path: nullcontext(np.zeros((1, 10, 10))),
    )
    monkeypatch.setattr(
        vesedge_cli,
        "load_extractor_from_module",
        lambda import_string: object(),
    )
    monkeypatch.setattr(vesedge_cli, "VesicleVideo", SuccessfulVideo)

    vesedge_cli.process_extract_file(path, _extract_args(tmp_path, path))

    assert observed["checkpoint"] == tmp_path / "sample.npz"
    assert observed["extraction_config"].pixels_per_micron == 1.0
    assert observed["extraction_config"].n_angular_samples is None
