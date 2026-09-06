"""Regression tests for VesEdge source-video provenance in the CLI."""

import argparse
from pathlib import Path

import numpy as np

from vesmod.VesEdge import ArrayFrameSource
from vesmod.cli import vesedge_cli


def test_process_extract_file_sets_source_path_on_video(tmp_path, monkeypatch):
    """Test CLI extraction preserves the original ND2 path on VesicleVideo."""
    observed = {}

    class FakeEdges:
        def save_checkpoint(self, path):
            observed["checkpoint"] = path

    class FakeVideo:
        def __init__(self, frames):
            observed["frames"] = frames
            self.source_path = None

        def extract_edges(self, extractor, extraction_config):
            observed["source_path"] = self.source_path
            return FakeEdges()

    path = tmp_path / "sample.nd2"
    monkeypatch.setattr(
        vesedge_cli,
        "open_frame_source",
        lambda input_path: ArrayFrameSource(
            np.zeros((1, 10, 10))
        ),
    )
    monkeypatch.setattr(
        vesedge_cli,
        "load_extractor_from_module",
        lambda import_string: object(),
    )
    monkeypatch.setattr(vesedge_cli, "VesicleVideo", FakeVideo)

    args = argparse.Namespace(
        input_path=path,
        output_dir=tmp_path / "checkpoints",
        no_gif=True,
        overwrite=True,
        extractor_file=None,
        extractor="vesmod.VesEdge:extract_edge_from_frame",
        extractor_name="extract_edge_from_frame",
        downsample=False,
        n_samples=120,
        pixels_per_micron=1.0,
    )

    vesedge_cli.process_extract_file(path, args)

    assert observed["source_path"] == Path(path)
    assert observed["checkpoint"] == tmp_path / "checkpoints" / "sample.npz"
