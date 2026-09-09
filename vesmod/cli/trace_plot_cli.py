"""CLI support for plotting selected VesEdge contours."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vesmod.VesEdge import (
    EdgeTracePlotConfig,
    VesicleEdges,
    load_recorded_qc,
    plot_edge_traces,
    replay_recorded_qc,
)
from vesmod.io import open_checkpoint_frames, resolve_source_path


def add_trace_plot_parser(subparsers) -> None:
    """Add the ``vesedge plot-traces`` parser."""
    parser = subparsers.add_parser(
        "plot-traces",
        help="Plot selected accepted VesEdge contours as a time overlay.",
    )
    parser.add_argument("checkpoint", type=Path, help="VesEdge .npz checkpoint.")
    parser.add_argument(
        "--qc-dir",
        type=Path,
        required=True,
        help="QC directory containing vesedge_qc.json.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        required=True,
        metavar="FRAME",
        help="Original source-video frame indices to overlay.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output figure path.",
    )
    parser.add_argument(
        "--background-frame",
        type=int,
        default=None,
        help="Source-video frame to use as an aligned image background.",
    )
    parser.add_argument(
        "--contour",
        choices=("full", "analysis"),
        default="full",
        help="Contour representation to draw. Default: full.",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap.")
    parser.add_argument("--linewidth", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Do not add the source-frame colorbar.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing figure and JSON sidecar.",
    )


def _load_edges_and_qc(
    checkpoint: Path,
    qc_dir: Path,
    *,
    source_path: Path | None,
):
    """Load a checkpoint and replay its recorded QC configuration."""
    edges = VesicleEdges.from_checkpoint(checkpoint)
    selection = load_recorded_qc(qc_dir, [checkpoint])
    if selection.requires_frames:
        if source_path is None:
            source_path = resolve_source_path(edges.source_path, checkpoint)
        with open_checkpoint_frames(source_path) as frames:
            replay_recorded_qc(edges, checkpoint, selection, frames=frames)
    else:
        replay_recorded_qc(edges, checkpoint, selection)
    return edges


def process_trace_plot(args: argparse.Namespace) -> tuple[Path, Path]:
    """Render one trace-overlay figure and its reproducibility sidecar."""
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    sidecar = output.with_suffix(".json")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if (output.exists() or sidecar.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists; use --overwrite: {output}"
        )

    source_path = None
    if args.background_frame is not None:
        provisional_edges = VesicleEdges.from_checkpoint(checkpoint)
        source_path = resolve_source_path(provisional_edges.source_path, checkpoint)
    edges = _load_edges_and_qc(
        checkpoint,
        args.qc_dir,
        source_path=source_path,
    )

    background = None
    if args.background_frame is not None:
        if args.background_frame < 0:
            raise ValueError("background frame must be nonnegative.")
        assert source_path is not None
        with open_checkpoint_frames(source_path) as frames:
            if args.background_frame >= len(frames):
                raise IndexError(
                    "background frame is outside the source-video range."
                )
            background = frames[args.background_frame]

    config = EdgeTracePlotConfig(
        centered=background is None,
        contour=args.contour,
        cmap=args.cmap,
        linewidth=args.linewidth,
        alpha=args.alpha,
        show_colorbar=not args.no_colorbar,
    )
    figure, _ = plot_edge_traces(
        edges,
        args.frames,
        background=background,
        config=config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    figure.clf()

    selected_frames = sorted(set(args.frames))
    metadata = {
        "checkpoint": str(checkpoint),
        "qc_provenance": str(args.qc_dir.expanduser().resolve()),
        "frame_indices": selected_frames,
        "background_frame": args.background_frame,
        "source_path": str(source_path) if source_path is not None else None,
        "plot_config": asdict(config),
        "pixels_per_micron": edges.extraction_config.pixels_per_micron,
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output, sidecar


def run_trace_plot(args: argparse.Namespace) -> int:
    """Run the trace-overlay command."""
    output, sidecar = process_trace_plot(args)
    print(f"Saved trace overlay: {output}")
    print(f"Saved trace-overlay metadata: {sidecar}")
    return 0
