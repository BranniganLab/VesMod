"""Tests for the experimental internal-structure CLI command."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from vesmod.VesEdge import (
    EdgeDetection,
    EdgeQCConfig,
    ImageContour,
    QCFlag,
)
from vesmod.VesEdge.experimental import InternalStructureRegion
from vesmod.cli import internal_structures_cli, vesedge_cli


def _args(tmp_path, checkpoint):
    """Return standard internal-structure CLI arguments."""
    return argparse.Namespace(
        input_path=checkpoint,
        recursive=False,
        output_dir=tmp_path / "output",
        video_root=None,
        qc_results=None,
        include_unqced=True,
        membrane_exclusion_px=5,
        background_sigma_px=8.0,
        threshold_sigma=4.0,
        min_region_area_px=9,
        light_grow_sigma=1.5,
        min_light_circularity=0.2,
        min_light_solidity=0.8,
        max_light_eccentricity=0.95,
        structure_boundary_exclusion_px=20,
        filament_seed_threshold=0.7,
        filament_grow_threshold=0.35,
        filament_scales_px=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        min_filament_length_px=20,
        bubble_edge_sigma=2.0,
        bubble_edge_grow_sigma=1.0,
        bubble_closing_px=4,
        min_bubble_area_px=100,
        min_bubble_boundary_fraction=0.45,
        min_bubble_circularity=0.2,
        min_bubble_solidity=0.8,
        max_bubble_eccentricity=0.95,
        max_bubble_area_fraction=0.5,
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
            "--include-unqced",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.command == "internal-structures"
    assert args.input_path == Path("checkpoints")
    assert args.output_dir == tmp_path
    assert args.save_masks
    assert args.background_sigma_px == pytest.approx(30.0)
    assert args.structure_boundary_exclusion_px == 20
    assert args.bubble_closing_px == 4
    assert args.filament_scales_px == pytest.approx(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )


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
        structure_count = 1
        light_area_fraction = 0.0
        dark_region_area_fraction = 0.0
        filament_area_fraction = 0.0
        filament_length_px = 0
        bubble_area_fraction = 0.0
        bubble_count = 0
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

        @staticmethod
        def to_full_frame_channel_mask(structure_type):
            assert structure_type in {
                "light_region",
                "dark_region",
                "dark_filament",
                "bubble",
            }
            return np.zeros((10, 10), dtype=bool)

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
        None,
    )

    with (args.output_dir / "sample_regions.csv").open() as region_file:
        region = next(csv.DictReader(region_file))
    masks = np.load(args.output_dir / "sample_masks.npz")

    assert region["polarity"] == "dark"
    assert region["structure_type"] == "unclassified"
    assert region["centroid_y"] == "6.5"
    assert region["centroid_x"] == "7.5"
    assert masks["frame_indices"].tolist() == [0]
    assert masks["structure_masks"][0, 5, 7]
    assert not masks["light_region_masks"].any()
    assert not masks["dark_filament_masks"].any()
    assert not masks["bubble_region_masks"].any()
    assert summary["median_area_fraction"] == 0.16


def test_process_checkpoint_reports_unreadable_video(tmp_path, monkeypatch):
    """Test unreadable source videos return a load-error summary row."""
    checkpoint = tmp_path / "sample.npz"
    checkpoint.touch()
    video_path = tmp_path / "sample.nd2"
    video_path.touch()

    class FakeEdges:
        source_path = video_path
        detections = []

    monkeypatch.setattr(
        internal_structures_cli.VesicleEdges,
        "from_checkpoint",
        lambda path: FakeEdges(),
    )
    monkeypatch.setattr(
        internal_structures_cli.nd2,
        "imread",
        lambda path: (_ for _ in ()).throw(OSError("truncated ND2")),
    )

    args = _args(tmp_path, checkpoint)
    summary = internal_structures_cli.process_checkpoint(
        checkpoint,
        args,
        internal_structures_cli.config_from_args(args),
        None,
    )

    assert summary["status"] == "load_error"
    assert summary["error"] == "truncated ND2"


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


def test_load_qc_selection_reconstructs_recorded_config(tmp_path):
    """Test frame eligibility comes from the selected QC provenance."""
    checkpoint = tmp_path / "checkpoints" / "sample.npz"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir()
    (qc_dir / "vesedge_qc.json").write_text(
        json.dumps(
            {
                "checkpoint_manifest": [str(checkpoint.resolve())],
                "qc_config": {
                    "curvature_threshold": 7.0,
                    "enable_curvature_qc": True,
                },
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, checkpoint)
    args.qc_results = qc_dir
    args.include_unqced = False

    config, provenance_path = internal_structures_cli._load_qc_selection(
        args,
        [checkpoint],
    )

    assert config.curvature.threshold == 7.0
    assert provenance_path == (qc_dir / "vesedge_qc.json").resolve()


def test_process_checkpoint_does_not_measure_qc_rejected_frame(
    tmp_path,
    monkeypatch,
):
    """Test rejected frames remain explicit and never reach the detector."""
    checkpoint = tmp_path / "sample.npz"
    checkpoint.touch()
    video_path = tmp_path / "sample.nd2"
    video_path.touch()
    contour = ImageContour((5.0, 5.0), np.full(12, 3.0))
    detection = EdgeDetection(contour, contour, frame_index=0)

    class FakeEdges:
        source_path = video_path
        detections = [detection]
        qc_result = None

        def run_qc(self, config):
            detection.qc.flags.add(next(iter(QCFlag)))
            self.qc_result = object()
            raise ValueError("no frames passed quality control")

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
        lambda frame, edge, config: pytest.fail(
            "QC-rejected frame reached internal-structure detection"
        ),
    )
    args = _args(tmp_path, checkpoint)
    args.include_unqced = False
    args.qc_results = tmp_path / "qc"

    summary = internal_structures_cli.process_checkpoint(
        checkpoint,
        args,
        internal_structures_cli.config_from_args(args),
        EdgeQCConfig(curvature_threshold=5.0),
    )

    with (args.output_dir / "sample_frames.csv").open() as frame_file:
        frame_row = next(csv.DictReader(frame_file))

    assert frame_row["status"] == "qc_rejected"
    assert summary["qc_rejected"] == 1
    assert summary["analyzed_frames"] == 0


def test_save_overlay_gif_uses_shared_qc_aware_renderer(tmp_path, monkeypatch):
    """Test internal-structure GIFs delegate rendering to VesicleVideo."""
    frames = np.zeros((1, 10, 10))
    edges = argparse.Namespace(source_path=tmp_path / "sample.nd2")
    result = argparse.Namespace(
        structured_area_fraction=0.2,
        structure_count=1,
        to_full_frame_mask=lambda: np.zeros((10, 10), dtype=bool),
    )
    observed = {}

    class FakeAxis:
        @staticmethod
        def imshow(*_, **__):
            return None

    class FakeVideo:
        def __init__(self, supplied_frames, source_path=None):
            observed["frames"] = supplied_frames
            observed["source_path"] = source_path

        def make_vesicle_gif(
            self,
            path,
            supplied_edges,
            frame_decorator=None,
            title_provider=None,
        ):
            observed["path"] = path
            observed["edges"] = supplied_edges
            observed["decorator_result"] = frame_decorator(FakeAxis(), 0)
            observed["title"] = title_provider(0)

    monkeypatch.setattr(internal_structures_cli, "VesicleVideo", FakeVideo)

    internal_structures_cli._save_overlay_gif(
        tmp_path / "sample",
        frames,
        edges,
        {0: result},
    )

    assert observed["frames"] is frames
    assert observed["source_path"] == edges.source_path
    assert observed["edges"] is edges
    assert observed["path"] == tmp_path / "sample_internal_structures.gif"
    assert observed["decorator_result"] is None
    assert observed["title"] == "frame 0: structured=0.200, regions=1"
