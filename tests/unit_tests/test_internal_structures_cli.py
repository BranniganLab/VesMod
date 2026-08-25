"""Tests for the experimental internal-structure CLI command."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import pytest

from vesmod.VesEdge import EdgeDetection, ImageContour, InternalStructureRegion
from vesmod.cli import internal_structures_cli, vesedge_cli


def _args(tmp_path, checkpoint):
    """Return standard internal-structure CLI arguments."""
    return argparse.Namespace(
        input_path=checkpoint,
        recursive=False,
        output_dir=tmp_path / "output",
        video_root=None,
        membrane_exclusion_px=5,
        background_sigma_px=8.0,
        threshold_sigma=4.0,
        min_region_area_px=9,
        save_masks=True,
        no_gif=True,
        overwrite=False,
    )


def test_parse_args_selects_internal_structures_subcommand(monkeypatch, tmp_path):
    """Test measurement arguments are scoped to their independent command."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "internal-structures",
            "checkpoints",
            "--output-dir",
            str(tmp_path),
            "--save-masks",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.command == "internal-structures"
    assert args.input_path == Path("checkpoints")
    assert args.output_dir == tmp_path
    assert args.save_masks


def test_process_checkpoint_writes_measurements_in_original_coordinates(
    tmp_path,
    monkeypatch,
):
    """Test the CLI preserves frame identity and original image coordinates."""
    checkpoint = tmp_path / "sample.npz"
    checkpoint.touch()
    video_path = tmp_path / "sample.nd2"
    video_path.touch()
    contour = ImageContour((5.0, 5.0), np.full(12, 3.0))

    class FakeEdges:
        source_path = video_path
        detections = [EdgeDetection(contour, contour, frame_index=0)]

    class FakeResult:
        usable_area_px = 25
        structured_area_px = 4
        structured_area_fraction = 0.16
        noise_sigma = 0.5
        regions = (
            InternalStructureRegion(
                label=1,
                area_px=4,
                centroid_yx=(6.5, 7.5),
                bbox_yx=(5, 6, 8, 9),
                mean_signed_residual=-3.0,
            ),
        )

        @staticmethod
        def to_full_frame_mask():
            mask = np.zeros((10, 10), dtype=bool)
            mask[5:7, 7:9] = True
            return mask

    monkeypatch.setattr(
        internal_structures_cli.VesicleEdges,
        "from_checkpoint",
        lambda path: FakeEdges(),
    )
    monkeypatch.setattr(
        internal_structures_cli.nd2,
        "imread",
        lambda path: np.zeros((1, 10, 10)),
    )
    monkeypatch.setattr(
        internal_structures_cli,
        "detect_internal_structures",
        lambda frame, edge, config: FakeResult(),
    )

    args = _args(tmp_path, checkpoint)
    summary = internal_structures_cli.process_checkpoint(
        checkpoint,
        args,
        internal_structures_cli.config_from_args(args),
    )

    with (args.output_dir / "sample_regions.csv").open() as region_file:
        region = next(csv.DictReader(region_file))
    masks = np.load(args.output_dir / "sample_masks.npz")

    assert region["polarity"] == "dark"
    assert region["centroid_y"] == "6.5"
    assert region["centroid_x"] == "7.5"
    assert masks["frame_indices"].tolist() == [0]
    assert masks["structure_masks"][0, 5, 7]
    assert summary["median_area_fraction"] == 0.16


def test_resolve_video_path_can_relocate_recorded_source(tmp_path):
    """Test --video-root can replace a stale checkpoint source directory."""
    video_root = tmp_path / "videos"
    video_root.mkdir()
    replacement = video_root / "sample.nd2"
    replacement.touch()

    resolved = internal_structures_cli._resolve_video_path(
        Path("/old/location/sample.nd2"),
        video_root,
        tmp_path / "checkpoints" / "sample.npz",
    )

    assert resolved == replacement.resolve()


def test_resolve_video_path_infers_legacy_checkpoint_sibling(tmp_path):
    """Test a legacy checkpoint can infer a same-stem neighboring ND2 file."""
    checkpoint = tmp_path / "sample.npz"
    checkpoint.touch()
    video = tmp_path / "sample.nd2"
    video.touch()

    resolved = internal_structures_cli._resolve_video_path(
        None,
        None,
        checkpoint,
    )

    assert resolved == video.resolve()


def test_resolve_video_path_infers_legacy_checkpoint_under_video_root(tmp_path):
    """Test --video-root supports checkpoints without stored provenance."""
    checkpoint = tmp_path / "checkpoints" / "sample.npz"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    nested_video_dir = tmp_path / "videos" / "nested"
    nested_video_dir.mkdir(parents=True)
    video = nested_video_dir / "sample.nd2"
    video.touch()

    resolved = internal_structures_cli._resolve_video_path(
        None,
        tmp_path / "videos",
        checkpoint,
    )

    assert resolved == video.resolve()


def test_resolve_video_path_rejects_ambiguous_legacy_matches(tmp_path):
    """Test source inference never silently chooses between duplicate names."""
    checkpoint = tmp_path / "checkpoints" / "sample.npz"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    video_root = tmp_path / "videos"
    for directory in (video_root / "a", video_root / "b"):
        directory.mkdir(parents=True)
        (directory / "sample.nd2").touch()

    with pytest.raises(ValueError, match="Multiple source videos match"):
        internal_structures_cli._resolve_video_path(
            None,
            video_root,
            checkpoint,
        )
