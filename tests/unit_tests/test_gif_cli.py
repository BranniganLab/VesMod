"""Tests for standalone VesEdge GIF generation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vesmod.VesEdge import EdgeQCConfig
from vesmod.cli import gif_cli, vesedge_cli


def _args(tmp_path, input_path, style="edges"):
    """Return standard GIF CLI arguments."""
    return argparse.Namespace(
        input_path=input_path,
        output_dir=tmp_path / "gifs",
        style=style,
        qc_dir=None,
        recursive=True,
        overwrite=False,
    )


def test_parse_args_selects_gif_subcommand(monkeypatch, tmp_path):
    """Test GIF options are registered under vesedge gif."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "gif",
            "checkpoints",
            "--output-dir",
            str(tmp_path),
            "--style",
            "original",
            "--recursive",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.command == "gif"
    assert args.style == "original"
    assert args.recursive
    assert args.output_dir == tmp_path


def test_checkpoint_paths_and_qc_pairing_preserve_recursive_structure(tmp_path):
    """Test equal stems in separate folders map to separate QC arrays."""
    checkpoints = tmp_path / "checkpoints"
    first = checkpoints / "a" / "sample.npz"
    second = checkpoints / "b" / "sample.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    qc_dir = tmp_path / "qc"

    selected = gif_cli._checkpoint_paths(checkpoints, recursive=True)

    assert selected == [first, second]
    assert gif_cli._paired_qc_path(
        first,
        checkpoints,
        qc_dir,
    ) == qc_dir / "a" / "sample.npy"
    assert gif_cli._paired_qc_path(
        second,
        checkpoints,
        qc_dir,
    ) == qc_dir / "b" / "sample.npy"


def test_load_qc_config_uses_recorded_provenance(tmp_path):
    """Test QC-colored rendering reuses the saved QC settings."""
    provenance = {
        "qc_config": {
            "curvature_threshold": 8.0,
            "enable_curvature_qc": True,
            "max_relative_area_deviation": 0.4,
            "enable_area_qc": False,
        }
    }
    (tmp_path / "vesedge_qc.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )

    config = gif_cli._load_qc_config(tmp_path)

    assert config.curvature_threshold == pytest.approx(8.0)
    assert config.max_relative_area_deviation == pytest.approx(0.4)
    assert not config.enable_area_qc


def test_apply_recorded_qc_verifies_paired_array(tmp_path):
    """Test QC state is reconstructed and checked against the paired .npy."""
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "nested" / "sample.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    qc_dir = tmp_path / "qc"
    qc_path = qc_dir / "nested" / "sample.npy"
    qc_path.parent.mkdir(parents=True)
    expected = np.ones((2, 4))
    np.save(qc_path, expected)
    config = EdgeQCConfig(curvature_threshold=5.0)

    class FakeEdges:
        accepted_radii_microns = expected
        qc_result = None

        def run_qc(self, supplied):
            assert supplied is config
            self.qc_result = object()

    gif_cli._apply_recorded_qc(
        FakeEdges(),
        checkpoint,
        checkpoint_root,
        qc_dir,
        config,
    )


@pytest.mark.parametrize(
    ("style", "expects_overlay"),
    [("original", False), ("edges", True)],
)
def test_process_gif_file_selects_annotation_style(
    tmp_path,
    monkeypatch,
    style,
    expects_overlay,
):
    """Test original and edge styles differ only in the supplied overlay."""
    checkpoint = tmp_path / "sample.npz"
    checkpoint.touch()
    source = tmp_path / "sample.npy"
    np.save(source, np.zeros((2, 5, 5)))
    fake_edges = argparse.Namespace(source_path=source)
    observed = {}

    class FakeVideo:
        def __init__(self, frames, source_path=None):
            observed["frames"] = frames
            observed["source_path"] = source_path

        def make_vesicle_gif(self, output_path, overlay):
            observed["output_path"] = output_path
            observed["overlay"] = overlay

    monkeypatch.setattr(
        gif_cli.VesicleEdges,
        "from_checkpoint",
        lambda path: fake_edges,
    )
    monkeypatch.setattr(gif_cli, "VesicleVideo", FakeVideo)
    args = _args(tmp_path, checkpoint, style=style)

    gif_cli.process_gif_file(checkpoint, args, qc_config=None)

    assert (observed["overlay"] is not None) is expects_overlay
    assert observed["output_path"] == tmp_path / "gifs" / "sample.gif"
    assert observed["source_path"] == source.resolve()
    assert observed["frames"].shape == (2, 5, 5)


def test_process_gif_file_reports_full_checkpoint_path(tmp_path, monkeypatch, capsys):
    """Test one failed rendering reports its exact checkpoint and returns."""
    checkpoint = tmp_path / "nested" / "sample.npz"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    class MissingSource:
        source_path = None

    monkeypatch.setattr(
        gif_cli.VesicleEdges,
        "from_checkpoint",
        lambda path: MissingSource(),
    )

    gif_cli.process_gif_file(
        checkpoint,
        _args(tmp_path, checkpoint),
        qc_config=None,
    )

    output = capsys.readouterr().out
    assert f"Failed to make GIF for {checkpoint.resolve()}" in output
    assert "Checkpoint does not record a source video path" in output


def test_run_gif_requires_qc_directory_for_qc_style(tmp_path):
    """Test QC-colored rendering cannot silently omit QC provenance."""
    args = _args(tmp_path, tmp_path / "sample.npz", style="qc")

    with pytest.raises(ValueError, match="--qc-dir is required"):
        gif_cli.run_gif(args)
