"""Standalone recursive GIF generation for VesEdge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vesmod.VesEdge import (
    EdgeQCConfig,
    FrameSource,
    VesicleEdges,
    VesicleVideo,
    open_frame_source,
)

from .input_selection import InputPathsAction, select_input_files


def add_gif_parser(subparsers) -> None:
    """Add the standalone GIF-generation subcommand."""
    parser = subparsers.add_parser(
        "gif",
        help="Render original, edge-overlay, or QC-colored GIFs.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="+",
        action=InputPathsAction,
        help="One or more VesEdge .npz files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated GIFs.",
    )
    parser.add_argument(
        "--style",
        choices=("original", "edges", "qc"),
        default="edges",
        help=(
            "Render unannotated frames, detected edges, or edges colored by "
            "QC acceptance. Default: edges."
        ),
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=None,
        help=(
            "QC output directory containing paired .npy files and "
            "vesedge_qc.json. Required with --style qc."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search checkpoint subdirectories and recursive glob matches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GIF outputs.",
    )


def _checkpoint_paths(
    input_path: Path | list[Path],
    recursive: bool,
) -> list[Path]:
    """Return checkpoints selected by explicit paths, directories, or globs."""
    checkpoints, _ = select_input_files(input_path, ".npz", recursive)
    return checkpoints


def _relative_checkpoint_path(checkpoint: Path, input_path: Path) -> Path:
    """Return a selected checkpoint path relative to its input root."""
    checkpoint = checkpoint.resolve()
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        return Path(checkpoint.name)
    return checkpoint.relative_to(input_path)


def _paired_qc_path(
    checkpoint: Path,
    input_path: Path,
    qc_dir: Path,
) -> Path:
    """Map one checkpoint to its QC array by relative path and stem."""
    relative = _relative_checkpoint_path(checkpoint, input_path)
    return qc_dir.expanduser().resolve() / relative.with_suffix(".npy")


def _load_qc_config(qc_dir: Path) -> EdgeQCConfig:
    """Load the exact QC configuration recorded for a QC output directory."""
    provenance_path = qc_dir.expanduser().resolve() / "vesedge_qc.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"QC provenance does not exist: {provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    try:
        config_data = provenance["qc_config"]
    except KeyError as error:
        raise ValueError(
            f"QC provenance has no qc_config: {provenance_path}"
        ) from error
    return EdgeQCConfig(**config_data)


def _resolve_source_path(edges: VesicleEdges, checkpoint: Path) -> Path:
    """Resolve a checkpoint's source video, including local-name fallback."""
    if edges.source_path is None:
        raise ValueError("Checkpoint does not record a source video path.")

    recorded = Path(edges.source_path).expanduser()
    candidates = [recorded]
    if not recorded.is_absolute():
        candidates.append(checkpoint.parent / recorded)
    candidates.append(checkpoint.parent / recorded.name)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    attempted = ", ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        f"Source video for {checkpoint.resolve()} was not found. "
        f"Tried: {attempted}"
    )


def _load_frames(source_path: Path) -> FrameSource:
    """Open ND2 or NumPy source frames without eagerly loading the video."""
    return open_frame_source(source_path)


def _apply_recorded_qc(
    edges: VesicleEdges,
    frames: FrameSource,
    checkpoint: Path,
    input_path: Path,
    qc_dir: Path,
    qc_config: EdgeQCConfig,
) -> None:
    """Reconstruct frame-level QC and verify the paired filtered output."""
    try:
        edges.run_qc(qc_config, frames)
    except ValueError:
        if edges.qc_result is None:
            raise
        if not edges.accepted_detections:
            return
        raise

    qc_path = _paired_qc_path(checkpoint, input_path, qc_dir)
    if not qc_path.is_file():
        raise FileNotFoundError(
            f"No paired QC .npy exists for {checkpoint.resolve()}: {qc_path}"
        )

    saved_radii = np.load(qc_path, allow_pickle=False)
    reconstructed = edges.accepted_radii_microns
    if saved_radii.shape != reconstructed.shape or not np.allclose(
        saved_radii,
        reconstructed,
        equal_nan=True,
    ):
        raise ValueError(
            f"Paired QC output does not match {checkpoint.resolve()}: {qc_path}"
        )


def process_gif_file(
    checkpoint: Path,
    args: argparse.Namespace,
    qc_config: EdgeQCConfig | None,
) -> None:
    """Render one checkpoint without aborting the surrounding batch."""
    relative = _relative_checkpoint_path(checkpoint, args.input_path)
    output_path = (
        args.output_dir.expanduser().resolve()
        / relative.with_suffix(".gif")
    )
    if output_path.exists() and not args.overwrite:
        print(f"Skipping {checkpoint.resolve()}: GIF already exists: {output_path}")
        return

    try:
        edges = VesicleEdges.from_checkpoint(checkpoint)
        source_path = _resolve_source_path(edges, checkpoint)
        with _load_frames(source_path) as frames:
            if args.style == "qc":
                _apply_recorded_qc(
                    edges,
                    frames,
                    checkpoint,
                    args.input_path,
                    args.qc_dir,
                    qc_config,
                )
            overlay = None if args.style == "original" else edges
            output_path.parent.mkdir(parents=True, exist_ok=True)
            VesicleVideo(frames, source_path=source_path).make_vesicle_gif(
                output_path,
                overlay,
            )
    except (FileNotFoundError, IndexError, OSError, ValueError) as error:
        print(f"Failed to make GIF for {checkpoint.resolve()}: {error}")
        return

    print(f"Saved GIF for {checkpoint.resolve()}: {output_path}")


def run_gif(args: argparse.Namespace) -> None:
    """Generate the selected GIF style for every selected checkpoint."""
    if args.style == "qc" and args.qc_dir is None:
        raise ValueError("--qc-dir is required with --style qc.")
    if args.style != "qc" and args.qc_dir is not None:
        raise ValueError("--qc-dir may only be used with --style qc.")

    checkpoints, input_root = select_input_files(
        args.input_path,
        ".npz",
        args.recursive,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No .npz checkpoints found for {args.input_path}")
    args.input_path = input_root

    args.output_dir = args.output_dir.expanduser().resolve()
    qc_config = (
        _load_qc_config(args.qc_dir)
        if args.style == "qc"
        else None
    )
    for checkpoint in checkpoints:
        process_gif_file(checkpoint, args, qc_config)
