"""Unit tests for the VesEdge command-line interface."""

import argparse
from pathlib import Path

import numpy as np

from vesmod.cli import vesedge_cli


def _args() -> argparse.Namespace:
    """Return standard CLI arguments for process_file tests."""
    return argparse.Namespace(
        no_gif=True,
        overwrite=True,
        extractor_file=None,
        extractor="vesmod.VesEdge:extract_edge_from_frame",
        extractor_name="extract_edge_from_frame",
        downsample=False,
        n_samples=120,
        pixels_per_micron=1.0,
    )


def test_process_file_reports_extraction_failure_and_returns(
    monkeypatch,
    capsys,
):
    """Test that a per-file extraction failure does not abort batch processing."""

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
    monkeypatch.setattr(
        vesedge_cli,
        "VesicleVideo",
        FailingVideo,
    )

    vesedge_cli.process_file(path, _args())

    captured = capsys.readouterr()
    assert (
        "Failed to extract edges from failed.nd2: no successful detections"
        in captured.out
    )


def test_process_file_saves_checkpoint_after_success(monkeypatch):
    """Test that successful extraction writes a reusable checkpoint."""
    observed = {}

    class FakeEdges:
        def save_checkpoint(self, path):
            observed["checkpoint"] = path

    class SuccessfulVideo:
        def __init__(self, frames):
            observed["frames"] = frames

        def extract_edges(self, extractor, extraction_config):
            observed["config"] = extraction_config
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
    monkeypatch.setattr(
        vesedge_cli,
        "VesicleVideo",
        SuccessfulVideo,
    )

    vesedge_cli.process_file(path, _args())

    assert observed["checkpoint"] == path
    assert observed["config"].pixels_per_micron == 1.0
    assert observed["config"].n_angular_samples is None
