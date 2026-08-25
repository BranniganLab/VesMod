"""CLI orchestration for experimental internal-structure measurements."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import nd2
import numpy as np

from vesmod.VesEdge import (
    EdgeDetection,
    InternalStructureConfig,
    InternalStructureFrameResult,
    VesicleEdges,
    detect_internal_structures,
    summarize_internal_structures,
)


def add_parser(subparsers) -> None:
    """Add the independent internal-structure analysis subcommand."""
    parser = subparsers.add_parser(
        "internal-structures",
        help="Measure resolvable structures inside extracted vesicle edges.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A VesEdge .npz checkpoint or directory containing checkpoints.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when input_path is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for measurements, diagnostic GIFs, and provenance.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help=(
            "Optional directory containing source videos that have moved since "
            "edge extraction. Videos are matched by checkpoint source filename."
        ),
    )
    parser.add_argument(
        "--membrane-exclusion-px",
        type=int,
        default=5,
        help="Pixels excluded inward from the detected membrane. Default: 5.",
    )
    parser.add_argument(
        "--background-sigma-px",
        type=float,
        default=8.0,
        help="Gaussian sigma for the smooth interior background. Default: 8.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=4.0,
        help="Absolute residual threshold in robust noise sigmas. Default: 4.",
    )
    parser.add_argument(
        "--min-region-area-px",
        type=int,
        default=9,
        help="Minimum retained connected-region area in pixels. Default: 9.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save compressed full-frame structure masks and frame indices.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Do not save a GIF overlaying detected internal structures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs from a different measurement configuration.",
    )


def config_from_args(args: argparse.Namespace) -> InternalStructureConfig:
    """Build internal-structure configuration from CLI arguments."""
    return InternalStructureConfig(
        membrane_exclusion_px=args.membrane_exclusion_px,
        background_sigma_px=args.background_sigma_px,
        threshold_sigma=args.threshold_sigma,
        min_region_area_px=args.min_region_area_px,
    )


def run(args: argparse.Namespace) -> None:
    """Measure internal structures for the selected checkpoints."""
    paths = _iter_checkpoints(args.input_path, args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {args.input_path}")

    config = config_from_args(args)
    _write_provenance(args, paths, config)
    summary_rows = [process_checkpoint(path, args, config) for path in paths]
    _write_csv(
        args.output_dir / "internal_structure_summary.csv",
        summary_rows,
        _SUMMARY_FIELDS,
    )


def process_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    config: InternalStructureConfig,
) -> dict:
    """Measure one checkpoint and write its frame- and region-level outputs."""
    relative_path = _relative_input_path(checkpoint_path, args.input_path)
    output_base = args.output_dir / relative_path.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    try:
        edges = VesicleEdges.from_checkpoint(checkpoint_path)
        video_path = _resolve_video_path(edges.source_path, args.video_root)
        frames = nd2.imread(video_path)
        if frames.ndim != 3:
            raise ValueError("Source video must contain a 3D frame array.")
        if frames.shape[0] != len(edges.detections):
            raise ValueError(
                "Source video frame count does not match the checkpoint: "
                f"{frames.shape[0]} != {len(edges.detections)}."
            )
    except (FileNotFoundError, IndexError, TypeError, ValueError) as error:
        message = str(error)
        print(f"Failed to analyze {checkpoint_path.name}: {message}")
        return _error_summary(relative_path, message)

    frame_rows = []
    region_rows = []
    results: dict[int, InternalStructureFrameResult] = {}
    for frame_index, (frame, edge_result) in enumerate(
        zip(frames, edges.detections, strict=True)
    ):
        if not isinstance(edge_result, EdgeDetection):
            frame_rows.append(
                _frame_error_row(frame_index, "extraction_failure", edge_result.error)
            )
            continue
        try:
            result = detect_internal_structures(
                frame,
                edge_result.full_contour,
                config,
            )
        except (TypeError, ValueError) as error:
            frame_rows.append(
                _frame_error_row(frame_index, "measurement_error", str(error))
            )
            continue

        results[frame_index] = result
        frame_rows.append(_frame_row(frame_index, result))
        region_rows.extend(_region_rows(frame_index, result))

    _write_csv(
        output_base.with_name(output_base.name + "_frames.csv"),
        frame_rows,
        _FRAME_FIELDS,
    )
    _write_csv(
        output_base.with_name(output_base.name + "_regions.csv"),
        region_rows,
        _REGION_FIELDS,
    )
    if args.save_masks:
        _save_masks(output_base, results, frames.shape[1:])
    if not args.no_gif:
        _save_overlay_gif(output_base, frames, edges, results)

    return _summary_row(relative_path, video_path, len(edges.detections), results)


def _resolve_video_path(
    stored_path: str | Path | None,
    video_root: Path | None,
) -> Path:
    """Resolve the source video recorded by an extraction checkpoint."""
    if stored_path is None:
        raise ValueError("Checkpoint does not record a source video path.")
    stored = Path(stored_path).expanduser()
    if stored.is_file():
        return stored.resolve()
    if video_root is not None:
        replacement = video_root.expanduser().resolve() / stored.name
        if replacement.is_file():
            return replacement
    raise FileNotFoundError(
        f"Source video does not exist: {stored}. Use --video-root if it moved."
    )


def _frame_row(
    frame_index: int,
    result: InternalStructureFrameResult,
) -> dict:
    """Return one successful frame measurement row."""
    return {
        "frame_index": frame_index,
        "usable_area_px": result.usable_area_px,
        "structured_area_px": result.structured_area_px,
        "structured_area_fraction": result.structured_area_fraction,
        "region_count": len(result.regions),
        "noise_sigma": result.noise_sigma,
        "status": "ok",
        "error": "",
    }


def _frame_error_row(frame_index: int, status: str, error: str) -> dict:
    """Return one unsuccessful frame measurement row."""
    return {
        "frame_index": frame_index,
        "usable_area_px": "",
        "structured_area_px": "",
        "structured_area_fraction": "",
        "region_count": "",
        "noise_sigma": "",
        "status": status,
        "error": error,
    }


def _region_rows(
    frame_index: int,
    result: InternalStructureFrameResult,
) -> list[dict]:
    """Return original-coordinate rows for all regions in one frame."""
    rows = []
    for region in result.regions:
        min_y, min_x, max_y, max_x = region.bbox_yx
        centroid_y, centroid_x = region.centroid_yx
        rows.append(
            {
                "frame_index": frame_index,
                "region_label": region.label,
                "polarity": region.polarity,
                "area_px": region.area_px,
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "bbox_min_y": min_y,
                "bbox_min_x": min_x,
                "bbox_max_y": max_y,
                "bbox_max_x": max_x,
                "mean_signed_residual": region.mean_signed_residual,
            }
        )
    return rows


def _summary_row(
    relative_path: Path,
    video_path: Path,
    frame_count: int,
    results: dict[int, InternalStructureFrameResult],
) -> dict:
    """Return population-segmentation inputs for one analyzed video."""
    base = {
        "file": str(relative_path),
        "source_video": str(video_path),
        "frames": frame_count,
        "analyzed_frames": len(results),
        "measurement_failures": frame_count - len(results),
        "status": "ok" if results else "no_analyzable_frames",
        "error": "",
    }
    if not results:
        return {
            **base,
            "median_area_fraction": "",
            "upper_area_fraction": "",
            "frame_prevalence": "",
        }
    summary = summarize_internal_structures(tuple(results.values()))
    return {
        **base,
        "median_area_fraction": summary.median_area_fraction,
        "upper_area_fraction": summary.upper_area_fraction,
        "frame_prevalence": summary.frame_prevalence,
    }


def _error_summary(relative_path: Path, error: str) -> dict:
    """Return a canonical row when a checkpoint cannot be analyzed."""
    return {
        "file": str(relative_path),
        "source_video": "",
        "frames": 0,
        "analyzed_frames": 0,
        "measurement_failures": 0,
        "median_area_fraction": "",
        "upper_area_fraction": "",
        "frame_prevalence": "",
        "status": "load_error",
        "error": error,
    }


def _save_masks(
    output_base: Path,
    results: dict[int, InternalStructureFrameResult],
    frame_shape: tuple[int, int],
) -> None:
    """Save compressed masks aligned with original video coordinates."""
    frame_indices = np.asarray(sorted(results), dtype=np.int64)
    if results:
        masks = np.stack(
            [results[index].to_full_frame_mask() for index in frame_indices]
        )
    else:
        masks = np.empty((0, *frame_shape), dtype=bool)
    path = output_base.with_name(output_base.name + "_masks.npz")
    np.savez_compressed(path, frame_indices=frame_indices, structure_masks=masks)


def _save_overlay_gif(
    output_base: Path,
    frames: np.ndarray,
    edges: VesicleEdges,
    results: dict[int, InternalStructureFrameResult],
) -> None:
    """Save a diagnostic GIF in original-image coordinates."""
    figure, axis = plt.subplots()

    def animate(frame_index: int) -> None:
        axis.clear()
        axis.imshow(frames[frame_index], cmap="gray")
        edge_result = edges.detections[frame_index]
        if isinstance(edge_result, EdgeDetection):
            axis.plot(
                edge_result.full_contour.x,
                edge_result.full_contour.y,
                color="tab:green",
                linewidth=1,
            )
        result = results.get(frame_index)
        if result is None:
            axis.set_title(f"frame {frame_index}: not analyzed")
            return
        overlay = np.ma.masked_where(
            ~result.to_full_frame_mask(),
            result.to_full_frame_mask(),
        )
        axis.imshow(overlay, cmap="autumn", alpha=0.55, vmin=0, vmax=1)
        axis.set_title(
            f"frame {frame_index}: "
            f"structured fraction={result.structured_area_fraction:.4f}"
        )

    animation = FuncAnimation(
        figure,
        animate,
        frames=frames.shape[0],
        interval=150,
        blit=False,
        repeat_delay=1000,
    )
    path = output_base.with_name(output_base.name + "_internal_structures.gif")
    animation.save(path)
    plt.close(figure)


def _write_provenance(
    args: argparse.Namespace,
    paths: list[Path],
    config: InternalStructureConfig,
) -> None:
    """Write batch provenance and reject accidental configuration mixing."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output_dir / "internal_structure_analysis.json"
    provenance = {
        "experimental_method": "internal_structures",
        "input_path": str(args.input_path.expanduser().resolve()),
        "recursive": args.recursive,
        "checkpoint_manifest": [str(path.resolve()) for path in paths],
        "config": asdict(config),
    }
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing != provenance and not args.overwrite:
            raise ValueError(
                "Output directory contains internal-structure results from a "
                "different input selection or configuration. Choose another "
                "--output-dir or use --overwrite."
            )
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write dictionaries to a CSV with stable columns, including if empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _iter_checkpoints(input_path: Path, recursive: bool) -> list[Path]:
    """Return selected checkpoint files."""
    resolved = input_path.expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.lower() != ".npz":
            raise ValueError(f"Expected a .npz file, got: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in resolved.glob(pattern)
        if path.is_file() and path.suffix.lower() == ".npz"
    )


def _relative_input_path(path: Path, input_path: Path) -> Path:
    """Return a checkpoint path relative to the selected input root."""
    resolved_path = path.expanduser().resolve()
    resolved_input = input_path.expanduser().resolve()
    if resolved_path == resolved_input:
        return Path(resolved_path.name)
    return resolved_path.relative_to(resolved_input)


_FRAME_FIELDS = [
    "frame_index",
    "usable_area_px",
    "structured_area_px",
    "structured_area_fraction",
    "region_count",
    "noise_sigma",
    "status",
    "error",
]

_REGION_FIELDS = [
    "frame_index",
    "region_label",
    "polarity",
    "area_px",
    "centroid_y",
    "centroid_x",
    "bbox_min_y",
    "bbox_min_x",
    "bbox_max_y",
    "bbox_max_x",
    "mean_signed_residual",
]

_SUMMARY_FIELDS = [
    "file",
    "source_video",
    "frames",
    "analyzed_frames",
    "measurement_failures",
    "median_area_fraction",
    "upper_area_fraction",
    "frame_prevalence",
    "status",
    "error",
]
