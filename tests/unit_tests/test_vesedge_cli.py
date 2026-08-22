"""Unit tests for the VesEdge command-line interface."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from vesmod.VesEdge import EdgeQCConfig
from vesmod.cli import vesedge_cli


def _extract_args(tmp_path=None) -> argparse.Namespace:
    """Return standard extraction CLI arguments."""
    return argparse.Namespace(
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


def _qc_args(tmp_path) -> argparse.Namespace:
    """Return standard QC CLI arguments."""
    return argparse.Namespace(
        output_dir=tmp_path,
        overwrite=True,
        curvature_threshold=5.0,
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        no_curvature_qc=False,
        no_population_qc=False,
    )


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

    vesedge_cli.process_extract_file(path, _extract_args())

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

    vesedge_cli.process_extract_file(path, _extract_args(tmp_path))

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

    edges = FakeEdges()
    monkeypatch.setattr(
        vesedge_cli.VesicleEdges,
        "from_checkpoint",
        lambda path: edges,
    )
    args = _qc_args(tmp_path)
    config = vesedge_cli._qc_config_from_args(args)

    row = vesedge_cli.process_qc_file(Path("sample.npz"), args, config)

    assert observed["npy"] == tmp_path / "sample.npy"
    assert observed["qc_config"] is config
    assert row["accepted"] == 2
    assert row["status"] == "ok"


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

    edges = FakeEdges()
    monkeypatch.setattr(
        vesedge_cli.VesicleEdges,
        "from_checkpoint",
        lambda path: edges,
    )
    args = _qc_args(tmp_path)
    config = vesedge_cli._qc_config_from_args(args)

    row = vesedge_cli.process_qc_file(Path("sample.npz"), args, config)

    assert row["accepted"] == 0
    assert row["status"] == "no_accepted_frames"
    assert "no frames passed quality control" in row["error"]


def test_write_qc_provenance_rejects_different_configuration(tmp_path):
    """Test a QC directory cannot silently mix configurations."""
    config = EdgeQCConfig(5.0, 10.0, 0.25)
    vesedge_cli._write_qc_provenance(
        tmp_path,
        config,
        Path("checkpoints"),
        overwrite=False,
    )

    different = EdgeQCConfig(8.0, 10.0, 0.25)
    with pytest.raises(ValueError, match="different input path or QC configuration"):
        vesedge_cli._write_qc_provenance(
            tmp_path,
            different,
            Path("checkpoints"),
            overwrite=False,
        )


def test_write_qc_provenance_records_configuration(tmp_path):
    """Test QC provenance contains the exact configuration used."""
    config = EdgeQCConfig(5.0, 10.0, 0.25)
    vesedge_cli._write_qc_provenance(
        tmp_path,
        config,
        Path("checkpoints"),
        overwrite=False,
    )

    data = json.loads((tmp_path / "vesedge_qc.json").read_text())
    assert data["qc_config"]["curvature_threshold"] == 5.0
    assert data["qc_config"]["enable_population_qc"] is True


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
