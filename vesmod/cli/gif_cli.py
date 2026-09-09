"""Standalone recursive GIF generation for VesEdge."""

from __future__ import annotations

import argparse
from pathlib import Path

from vesmod.VesEdge import (
    FrameSource,
    RecordedQCSelection,
    VesicleEdges,
    VesicleVideo,
    load_recorded_qc,
    replay_recorded_qc,
)

from vesmod.io import (
    map_output_path,
    open_checkpoint_frames,
    resolve_source_path,
)
from vesmod.cli.batch_policy import add_batch_policy_argument, exit_code, report_batch_summary
from vesmod.cli.input_selection import InputPathsAction, select_input_files


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
    add_batch_policy_argument(parser)


def _checkpoint_paths(
    input_path: Path | list[Path],
    recursive: bool,
) -> list[Path]:
    """Return checkpoints selected by explicit paths, directories, or globs."""
    checkpoints, _ = select_input_files(input_path, ".npz", recursive)
    return checkpoints


def _apply_recorded_qc(
    edges: VesicleEdges,
    frames: FrameSource,
    checkpoint: Path,
    input_path: Path,
    selection: RecordedQCSelection,
) -> None:
    """Reconstruct frame-level QC and verify the paired filtered output."""
    replay_recorded_qc(
        edges,
        checkpoint,
        selection,
        frames=frames,
        input_root=input_path,
        verify_paired_output=True,
    )


def process_gif_file(
    checkpoint: Path,
    args: argparse.Namespace,
    qc_selection: RecordedQCSelection | None,
) -> None:
    """Render one checkpoint without aborting the surrounding batch."""
    output_path = map_output_path(
        checkpoint, args.input_path, args.output_dir, suffix=".gif"
    )
    if output_path.exists() and not args.overwrite:
        print(f"Skipping {checkpoint.resolve()}: GIF already exists: {output_path}")
        return False

    try:
        edges = VesicleEdges.from_checkpoint(checkpoint)
        source_path = resolve_source_path(edges.source_path, checkpoint)
        with open_checkpoint_frames(source_path) as frames:
            if args.style == "qc":
                _apply_recorded_qc(
                    edges,
                    frames,
                    checkpoint,
                    args.input_path,
                    qc_selection,
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
    return True


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
    qc_selection = (
        load_recorded_qc(args.qc_dir, checkpoints)
        if args.style == "qc"
        else None
    )
    succeeded = skipped = failed = 0
    for checkpoint in checkpoints:
        result = process_gif_file(checkpoint, args, qc_selection)
        if result is True:
            succeeded += 1
        elif result is False:
            failed += 1
        else:
            skipped += 1
        if result is False and args.error_policy == "fail-fast":
            break
    report_batch_summary(succeeded + skipped + failed, succeeded, skipped, failed)
    return exit_code(failed, succeeded + skipped)
