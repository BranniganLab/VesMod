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
        curvature_threshold=5.0,
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        no_population_qc=False,
    )


def test_process_file_reports_extraction_failure_and_returns(
    monkeypatch,
    capsys,
):
    """Test that a per-file extraction failure does not abort batch processing."""

    class FailingVideo:
        def __init__(self, frames, extraction_config, qc_config):
            pass

        def extract_edges(self, extractor):
            raise ValueError("no frames passed quality control")

        def make_vesicle_gif(self, path):
            raise AssertionError("GIF output should not be written after failure")

        def save_edge_to_npy(self, path):
            raise AssertionError("NumPy output should not be written after failure")

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
        "Failed to extract edges from failed.nd2: "
        "no frames passed quality control"
        in captured.out
    )
