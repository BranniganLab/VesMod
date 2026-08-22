"""Unit tests for the VesEdge command-line interface."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vesmod.VesEdge import EdgeQCConfig
from vesmod.cli import vesedge_cli


def _extract_args(tmp_path=None, input_path=Path("sample.nd2")) -> argparse.Namespace:
    """Return standard extraction CLI arguments."""
    return argparse.Namespace(
        input_path=input_path,
        output_dir=tmp_path,
        no_gif=True,
        overwrite=True,
        extractor_file=None,
        extractor="vesmod.VesEdge:extract_edge_from_frame",
        extractor_name="extract_edge_from_frame",
        downsample=False,
        n_samples=120,
        pixels_per_micron=1.0,
    )


def _qc_args(tmp_path, input_path=Path("sample.npz")) -> argparse.Namespace:
    """Return standard QC CLI arguments."""
    return argparse.Namespace(
        input_path=input_path,
        output_dir=tmp_path,
        recursive=False,
        overwrite=True,
        curvature_threshold=5.0,
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        no_curvature_qc=False,
        no_population_qc=False,
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


def test_parse_args_selects_qc_subcommand(monkeypatch, tmp_path):
    """Test QC arguments are scoped to the qc subcommand."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "qc",
            "checkpoints",
            "--output-dir",
            str(tmp_path),
            "--no-population-qc",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.command == "qc"
    assert args.input_path == Path("checkpoints")
    assert args.output_dir == tmp_path
    assert args.no_population_qc


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
        vesedge_cli.nd2,
        "imread",
        lambda input_path: np.zeros((1, 10, 10)),
    )
    monkeypatch.setattr(
        vesedge_cli,
        "load_extractor_from_module",
        lambda import_string: object(),
    )
    monkeypatch.setattr(vesedge_cli, "VesicleVideo", FailingVideo)

    vesedge_cli.process_extract_file(path, _extract_args(input_path=path))

    captured = capsys.readouterr()
    assert "Failed to extract failed.nd2: no successful detections" in captured.out


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
        vesedge_cli.nd2,
        "imread",
        lambda input_path: np.zeros((1, 10, 10)),
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


def test_process_qc_file_runs_qc_and_saves_filtered_output(tmp_path, monkeypatch):
    """Test QC loads a checkpoint, applies QC, and writes accepted contours."""
    observed = {}

    class Detection:
        def __init__(self):
            self.qc = argparse.Namespace(flags=set(), passed=True)

    class FakeEdges:
        def __init__(self):
            self.detections = [Detection(), Detection()]
            self.successful_detections = self.detections
            self.qc_result = None

        def run_qc(self, config):
            observed["qc_config"] = config
            self.qc_result = object()

        def save_edge_to_npy(self, path):
            observed["npy"] = path

    path = Path("sample.npz")
    edges = FakeEdges()
    monkeypatch.setattr(
        vesedge_cli.VesicleEdges,
        "from_checkpoint",
        lambda checkpoint_path: edges,
    )
    args = _qc_args(tmp_path, path)
    config = vesedge_cli._qc_config_from_args(args)

    row = vesedge_cli.process_qc_file(path, args, config)

    assert observed["npy"] == tmp_path / "sample.npy"
    assert observed["qc_config"] is config
    assert row["accepted"] == 2
    assert row["status"] == "ok"


def test_process_qc_file_preserves_relative_output_path(tmp_path, monkeypatch):
    """Test recursive QC outputs preserve checkpoint subdirectories."""
    observed = {}

    class Detection:
        def __init__(self):
            self.qc = argparse.Namespace(flags=set(), passed=True)

    class FakeEdges:
        detections = [Detection()]
        successful_detections = detections
        qc_result = object()

        def run_qc(self, config):
            pass

        def save_edge_to_npy(self, path):
            observed["npy"] = path

    input_root = tmp_path / "checkpoints"
    path = input_root / "nested" / "sample.npz"
    output_dir = tmp_path / "qc"
    monkeypatch.setattr(
        vesedge_cli.VesicleEdges,
        "from_checkpoint",
        lambda checkpoint_path: FakeEdges(),
    )
    args = _qc_args(output_dir, input_root)
    config = vesedge_cli._qc_config_from_args(args)

    vesedge_cli.process_qc_file(path, args, config)

    assert observed["npy"] == output_dir / "nested" / "sample.npy"


def test_process_qc_file_records_zero_accepted_frames(tmp_path, monkeypatch):
    """Test a completed QC run with no accepted frames is summarized."""

    class Detection:
        def __init__(self):
            self.qc = argparse.Namespace(flags=set(), passed=False)

    class FakeEdges:
        def __init__(self):
            self.detections = [Detection()]
            self.successful_detections = self.detections
            self.qc_result = None

        def run_qc(self, config):
            self.qc_result = object()
            raise ValueError("no frames passed quality control")

        def save_edge_to_npy(self, path):
            raise AssertionError("No .npy should be saved")

    path = Path("sample.npz")
    edges = FakeEdges()
    monkeypatch.setattr(
        vesedge_cli.VesicleEdges,
        "from_checkpoint",
        lambda checkpoint_path: edges,
    )
    args = _qc_args(tmp_path, path)
    config = vesedge_cli._qc_config_from_args(args)

    row = vesedge_cli.process_qc_file(path, args, config)

    assert row["accepted"] == 0
    assert row["status"] == "no_accepted_frames"
    assert "no frames passed quality control" in row["error"]


def test_process_qc_file_returns_load_error_summary(tmp_path, monkeypatch):
    """Test checkpoint load failures still produce canonical summary rows."""
    path = Path("broken.npz")
    monkeypatch.setattr(
        vesedge_cli.VesicleEdges,
        "from_checkpoint",
        lambda checkpoint_path: (_ for _ in ()).throw(ValueError("bad checkpoint")),
    )
    args = _qc_args(tmp_path, path)
    config = vesedge_cli._qc_config_from_args(args)

    row = vesedge_cli.process_qc_file(path, args, config)

    assert row == {
        "file": "broken.npz",
        "frames": 0,
        "successful_detections": 0,
        "extraction_failures": 0,
        "curvature_rejected": 0,
        "population_rejected": 0,
        "accepted": 0,
        "accepted_fraction": 0.0,
        "status": "load_error",
        "error": "bad checkpoint",
    }


def test_write_qc_provenance_rejects_different_configuration(tmp_path):
    """Test a QC directory cannot silently mix configurations."""
    checkpoint = tmp_path / "sample.npz"
    config = EdgeQCConfig(5.0, 10.0, 0.25)
    vesedge_cli._write_qc_provenance(
        tmp_path,
        config,
        checkpoint,
        False,
        [checkpoint],
        overwrite=False,
    )

    different = EdgeQCConfig(8.0, 10.0, 0.25)
    with pytest.raises(ValueError, match="different input selection or QC configuration"):
        vesedge_cli._write_qc_provenance(
            tmp_path,
            different,
            checkpoint,
            False,
            [checkpoint],
            overwrite=False,
        )


def test_write_qc_provenance_records_manifest_and_recursive_setting(tmp_path):
    """Test provenance identifies the exact resolved checkpoint selection."""
    input_root = tmp_path / "checkpoints"
    first = input_root / "a.npz"
    second = input_root / "nested" / "b.npz"
    config = EdgeQCConfig(5.0, 10.0, 0.25)

    vesedge_cli._write_qc_provenance(
        tmp_path / "qc",
        config,
        input_root,
        True,
        [first, second],
        overwrite=False,
    )

    data = json.loads((tmp_path / "qc" / "vesedge_qc.json").read_text())
    assert data["recursive"] is True
    assert data["checkpoint_manifest"] == [str(first.resolve()), str(second.resolve())]
    assert data["qc_config"]["curvature_threshold"] == 5.0


def test_overwrite_incompatible_provenance_removes_stale_outputs(tmp_path):
    """Test incompatible overwrite clears prior managed QC artifacts."""
    output_dir = tmp_path / "qc"
    checkpoint = tmp_path / "sample.npz"
    config = EdgeQCConfig(5.0, 10.0, 0.25)
    vesedge_cli._write_qc_provenance(
        output_dir,
        config,
        checkpoint,
        False,
        [checkpoint],
        overwrite=False,
    )
    stale = output_dir / "nested" / "orphan.npy"
    stale.parent.mkdir(parents=True)
    stale.touch()
    (output_dir / "qc_summary.csv").write_text("old")

    different = EdgeQCConfig(8.0, 10.0, 0.25)
    vesedge_cli._write_qc_provenance(
        output_dir,
        different,
        checkpoint,
        False,
        [checkpoint],
        overwrite=True,
    )

    assert not stale.exists()
    assert not (output_dir / "qc_summary.csv").exists()
    data = json.loads((output_dir / "vesedge_qc.json").read_text())
    assert data["qc_config"]["curvature_threshold"] == 8.0


def test_write_qc_summary_writes_batch_csv(tmp_path):
    """Test QC summary output contains one row per processed checkpoint."""
    rows = [
        {
            "file": "sample.npz",
            "frames": 10,
            "successful_detections": 9,
            "extraction_failures": 1,
            "curvature_rejected": 2,
            "population_rejected": 1,
            "accepted": 6,
            "accepted_fraction": 2 / 3,
            "status": "ok",
            "error": "",
        }
    ]

    vesedge_cli._write_qc_summary(tmp_path, rows)

    summary = (tmp_path / "qc_summary.csv").read_text()
    assert "sample.npz" in summary
    assert "curvature_rejected" in summary
    assert ",6," in summary


def test_run_qc_writes_summary_when_every_checkpoint_fails_to_load(
    tmp_path,
    monkeypatch,
):
    """Test an all-load-failure batch still writes qc_summary.csv."""
    input_dir = tmp_path / "checkpoints"
    input_dir.mkdir()
    first = input_dir / "first.npz"
    second = input_dir / "second.npz"
    first.touch()
    second.touch()
    output_dir = tmp_path / "qc"
    args = _qc_args(output_dir, input_dir)

    monkeypatch.setattr(
        vesedge_cli.VesicleEdges,
        "from_checkpoint",
        lambda checkpoint_path: (_ for _ in ()).throw(ValueError("bad checkpoint")),
    )

    vesedge_cli._run_qc(args)

    summary = (output_dir / "qc_summary.csv").read_text()
    assert "first.npz" in summary
    assert "second.npz" in summary
    assert summary.count("load_error") == 2
