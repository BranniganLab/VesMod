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
    EdgeQCConfig,
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
    frame_selection = parser.add_mutually_exclusive_group(required=True)
    frame_selection.add_argument(
        "--qc-results",
        type=Path,
        help=(
            "QC output directory, or its vesedge_qc.json file. Reapply that "
            "recorded configuration and analyze only passing frames."
        ),
    )
    frame_selection.add_argument(
        "--include-unqced",
        action="store_true",
        help=(
            "Analyze every successful edge detection without QC filtering. "
            "Intended for experimental method development."
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
        default=30.0,
        help="Gaussian sigma for the smooth interior background. Default: 30.",
    )
    parser.add_argument(
        "--threshold-sigma",
        type=float,
        default=4.0,
        help="Light-region seed threshold in robust noise sigmas. Default: 4.",
    )
    parser.add_argument(
        "--min-region-area-px",
        type=int,
        default=9,
        help="Minimum retained connected-region area in pixels. Default: 9.",
    )
    parser.add_argument(
        "--light-grow-sigma",
        type=float,
        default=1.5,
        help="Lower residual threshold used to grow seeded light regions.",
    )
    parser.add_argument(
        "--filament-threshold-sigma",
        type=float,
        default=1.5,
        help="Minimum multiscale dark-ridge response. Default: 1.5.",
    )
    parser.add_argument(
        "--filament-scales-px",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 3.0),
        help="Dark-filament ridge widths evaluated in pixels.",
    )
    parser.add_argument(
        "--min-filament-length-px",
        type=int,
        default=8,
        help="Minimum skeleton length retained as a filament. Default: 8.",
    )
    parser.add_argument(
        "--bubble-edge-sigma",
        type=float,
        default=2.0,
        help="Dark residual threshold used for bubble boundaries.",
    )
    parser.add_argument(
        "--bubble-closing-px",
        type=int,
        default=2,
        help="Maximum local gap closed in candidate bubble edges.",
    )
    parser.add_argument(
        "--min-bubble-area-px",
        type=int,
        default=25,
        help="Minimum area enclosed by a detected bubble. Default: 25.",
    )
    parser.add_argument(
        "--min-bubble-boundary-fraction",
        type=float,
        default=0.45,
        help="Minimum fraction of an enclosed boundary supported by dark pixels.",
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
        help="Replace outputs from a different measurement configuration.",
    )


def config_from_args(args: argparse.Namespace) -> InternalStructureConfig:
    """Build internal-structure configuration from CLI arguments."""
    return InternalStructureConfig(
        membrane_exclusion_px=args.membrane_exclusion_px,
        background_sigma_px=args.background_sigma_px,
        threshold_sigma=args.threshold_sigma,
        min_region_area_px=args.min_region_area_px,
        light_grow_sigma=getattr(args, "light_grow_sigma", 1.5),
        filament_threshold_sigma=getattr(
            args,
            "filament_threshold_sigma",
            1.5,
        ),
        filament_scales_px=tuple(
            getattr(args, "filament_scales_px", (1.0, 2.0, 3.0))
        ),
        min_filament_length_px=getattr(args, "min_filament_length_px", 8),
        bubble_edge_sigma=getattr(args, "bubble_edge_sigma", 2.0),
        bubble_closing_px=getattr(args, "bubble_closing_px", 2),
        min_bubble_area_px=getattr(args, "min_bubble_area_px", 25),
        min_bubble_boundary_fraction=getattr(
            args,
            "min_bubble_boundary_fraction",
            0.45,
        ),
    )


def run(args: argparse.Namespace) -> None:
    """Measure internal structures for the selected checkpoints."""
    _validate_input_output_paths(args.input_path, args.output_dir)
    paths = _iter_checkpoints(args.input_path, args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {args.input_path}")

    config = config_from_args(args)
    qc_config, qc_provenance_path = _load_qc_selection(args, paths)
    _write_provenance(
        args,
        paths,
        config,
        qc_config,
        qc_provenance_path,
    )
    summary_rows = [
        process_checkpoint(path, args, config, qc_config)
        for path in paths
    ]
    _write_csv(
        args.output_dir / "internal_structure_summary.csv",
        summary_rows,
        _SUMMARY_FIELDS,
    )


def process_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    config: InternalStructureConfig,
    qc_config: EdgeQCConfig | None,
) -> dict:
    """Measure one checkpoint and write its frame- and region-level outputs."""
    relative_path = _relative_input_path(checkpoint_path, args.input_path)
    output_base = args.output_dir / relative_path.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    try:
        edges = VesicleEdges.from_checkpoint(checkpoint_path)
        video_path = _resolve_video_path(
            edges.source_path,
            args.video_root,
            checkpoint_path,
        )
        frames = nd2.imread(video_path)
        if frames.ndim != 3:
            raise ValueError("Source video must contain a 3D frame array.")
        if frames.shape[0] != len(edges.detections):
            raise ValueError(
                "Source video frame count does not match the checkpoint: "
                f"{frames.shape[0]} != {len(edges.detections)}."
            )
        if qc_config is not None:
            _apply_qc(edges, qc_config)
    except (FileNotFoundError, IndexError, TypeError, ValueError) as error:
        message = str(error)
        print(f"Failed to analyze {_display_path(checkpoint_path)}: {message}")
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
        if qc_config is not None and not edge_result.qc.passed:
            frame_rows.append(
                _frame_error_row(frame_index, "qc_rejected", "")
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

    return _summary_row(
        relative_path,
        video_path,
        len(edges.detections),
        results,
        frame_rows,
    )



def _load_qc_selection(
    args: argparse.Namespace,
    checkpoint_paths: list[Path],
) -> tuple[EdgeQCConfig | None, Path | None]:
    """Load and validate the QC configuration selecting eligible frames."""
    if args.include_unqced:
        return None, None

    provenance_path = args.qc_results.expanduser().resolve()
    if provenance_path.is_dir():
        provenance_path = provenance_path / "vesedge_qc.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"QC provenance does not exist: {provenance_path}"
        )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    try:
        qc_config = EdgeQCConfig(**provenance["qc_config"])
        manifest = {
            str(Path(path).expanduser().resolve())
            for path in provenance["checkpoint_manifest"]
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid VesEdge QC provenance: {provenance_path}"
        ) from error

    unselected = [
        path
        for path in checkpoint_paths
        if str(path.resolve()) not in manifest
    ]
    if unselected:
        names = ", ".join(str(path) for path in unselected)
        raise ValueError(
            "Selected checkpoint(s) are not present in the QC manifest: "
            f"{names}"
        )
    return qc_config, provenance_path


def _apply_qc(edges: VesicleEdges, qc_config: EdgeQCConfig) -> None:
    """Apply frame eligibility while allowing a result with zero passing frames."""
    try:
        edges.run_qc(qc_config)
    except ValueError:
        if edges.qc_result is None:
            raise


def _resolve_video_path(
    stored_path: str | Path | None,
    video_root: Path | None,
    checkpoint_path: Path,
) -> Path:
    """Resolve a source video from provenance or an unambiguous filename."""
    if stored_path is not None:
        stored = Path(stored_path).expanduser()
        if stored.is_file():
            return stored.resolve()
        video_name = stored.name
    else:
        video_name = checkpoint_path.with_suffix(".nd2").name

    search_roots = [checkpoint_path.expanduser().resolve().parent]
    if video_root is not None:
        resolved_root = video_root.expanduser().resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(
                f"Video root does not exist or is not a directory: {resolved_root}"
            )
        if resolved_root not in search_roots:
            search_roots.append(resolved_root)

    matches = _find_video_matches(video_name, search_roots)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        match_list = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple source videos match {video_name}: {match_list}"
        )

    if stored_path is None:
        raise FileNotFoundError(
            "Checkpoint does not record a source video path and no matching "
            f"{video_name} was found beside it or under --video-root."
        )
    raise FileNotFoundError(
        f"Source video does not exist: {stored_path}. No matching {video_name} "
        "was found beside the checkpoint or under --video-root."
    )


def _find_video_matches(
    video_name: str,
    search_roots: list[Path],
) -> list[Path]:
    """Find unique case-insensitive filename matches below selected roots."""
    matches: set[Path] = set()
    lowercase_name = video_name.lower()
    for root in search_roots:
        for candidate in root.rglob("*"):
            if (
                candidate.is_file()
                and candidate.name.lower() == lowercase_name
            ):
                matches.add(candidate.resolve())
    return sorted(matches)


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
        "light_area_fraction": getattr(result, "light_area_fraction", 0.0),
        "filament_area_fraction": getattr(
            result,
            "filament_area_fraction",
            0.0,
        ),
        "filament_length_px": getattr(result, "filament_length_px", 0),
        "bubble_area_fraction": getattr(result, "bubble_area_fraction", 0.0),
        "bubble_count": getattr(result, "bubble_count", 0),
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
        "light_area_fraction": "",
        "filament_area_fraction": "",
        "filament_length_px": "",
        "bubble_area_fraction": "",
        "bubble_count": "",
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
                "structure_type": region.structure_type,
                "polarity": region.polarity,
                "area_px": region.area_px,
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "bbox_min_y": min_y,
                "bbox_min_x": min_x,
                "bbox_max_y": max_y,
                "bbox_max_x": max_x,
                "mean_signed_residual": region.mean_signed_residual,
                "skeleton_length_px": region.skeleton_length_px,
            }
        )
    return rows


def _summary_row(
    relative_path: Path,
    video_path: Path,
    frame_count: int,
    results: dict[int, InternalStructureFrameResult],
    frame_rows: list[dict],
) -> dict:
    """Return population-segmentation inputs for one analyzed video."""
    statuses = [row["status"] for row in frame_rows]
    base = {
        "file": str(relative_path),
        "source_video": str(video_path),
        "frames": frame_count,
        "analyzed_frames": len(results),
        "extraction_failures": statuses.count("extraction_failure"),
        "qc_rejected": statuses.count("qc_rejected"),
        "measurement_failures": statuses.count("measurement_error"),
        "status": "ok" if results else "no_analyzable_frames",
        "error": "",
    }
    if not results:
        return {
            **base,
            "median_area_fraction": "",
            "upper_area_fraction": "",
            "frame_prevalence": "",
            "median_light_area_fraction": "",
            "median_filament_area_fraction": "",
            "median_filament_length_px": "",
            "median_bubble_area_fraction": "",
            "median_bubble_count": "",
        }
    summary = summarize_internal_structures(tuple(results.values()))
    return {
        **base,
        "median_area_fraction": summary.median_area_fraction,
        "upper_area_fraction": summary.upper_area_fraction,
        "frame_prevalence": summary.frame_prevalence,
        "median_light_area_fraction": summary.median_light_area_fraction,
        "median_filament_area_fraction": summary.median_filament_area_fraction,
        "median_filament_length_px": summary.median_filament_length_px,
        "median_bubble_area_fraction": summary.median_bubble_area_fraction,
        "median_bubble_count": summary.median_bubble_count,
    }


def _error_summary(relative_path: Path, error: str) -> dict:
    """Return a canonical row when a checkpoint cannot be analyzed."""
    return {
        "file": str(relative_path),
        "source_video": "",
        "frames": 0,
        "analyzed_frames": 0,
        "extraction_failures": 0,
        "qc_rejected": 0,
        "measurement_failures": 0,
        "median_area_fraction": "",
        "upper_area_fraction": "",
        "frame_prevalence": "",
        "median_light_area_fraction": "",
        "median_filament_area_fraction": "",
        "median_filament_length_px": "",
        "median_bubble_area_fraction": "",
        "median_bubble_count": "",
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
        light_masks = np.stack(
            [
                _to_full_frame_channel_mask(
                    results[index],
                    getattr(results[index], "light_region_mask", None),
                )
                for index in frame_indices
            ]
        )
        filament_masks = np.stack(
            [
                _to_full_frame_channel_mask(
                    results[index],
                    getattr(results[index], "dark_filament_mask", None),
                )
                for index in frame_indices
            ]
        )
        bubble_masks = np.stack(
            [
                _to_full_frame_channel_mask(
                    results[index],
                    getattr(results[index], "bubble_region_mask", None),
                )
                for index in frame_indices
            ]
        )
    else:
        masks = np.empty((0, *frame_shape), dtype=bool)
        light_masks = masks.copy()
        filament_masks = masks.copy()
        bubble_masks = masks.copy()
    path = output_base.with_name(output_base.name + "_masks.npz")
    np.savez_compressed(
        path,
        frame_indices=frame_indices,
        structure_masks=masks,
        light_region_masks=light_masks,
        dark_filament_masks=filament_masks,
        bubble_region_masks=bubble_masks,
    )


def _to_full_frame_channel_mask(
    result: InternalStructureFrameResult,
    channel_mask: np.ndarray | None,
) -> np.ndarray:
    """Map one cropped channel mask into original-image coordinates."""
    original_shape = getattr(result, "original_shape", None)
    if original_shape is None:
        original_shape = result.to_full_frame_mask().shape
    full_mask = np.zeros(original_shape, dtype=bool)
    if channel_mask is None:
        return full_mask
    y_start, x_start = result.crop_origin_yx
    y_stop = y_start + channel_mask.shape[0]
    x_stop = x_start + channel_mask.shape[1]
    full_mask[y_start:y_stop, x_start:x_stop] = channel_mask
    return full_mask


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
        channel_specs = (
            (result.light_region_mask, "autumn"),
            (result.dark_filament_mask, "winter"),
            (result.bubble_region_mask, "cool"),
        )
        for channel_mask, color_map in channel_specs:
            full_mask = _to_full_frame_channel_mask(result, channel_mask)
            overlay = np.ma.masked_where(~full_mask, full_mask)
            axis.imshow(
                overlay,
                cmap=color_map,
                alpha=0.55,
                vmin=0,
                vmax=1,
            )
        axis.set_title(
            f"frame {frame_index}: "
            f"light={result.light_area_fraction:.3f}, "
            f"filament={result.filament_length_px}px, "
            f"bubbles={result.bubble_count}"
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
    qc_config: EdgeQCConfig | None,
    qc_provenance_path: Path | None,
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
        "frame_selection": (
            {"mode": "include_unqced"}
            if qc_config is None
            else {
                "mode": "qc",
                "qc_provenance": str(qc_provenance_path),
                "qc_config": asdict(qc_config),
            }
        ),
    }
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing != provenance:
            if not args.overwrite:
                raise ValueError(
                    "Output directory contains internal-structure results from "
                    "a different input selection or configuration. Choose "
                    "another --output-dir or use --overwrite."
                )
            _remove_managed_outputs(args.output_dir)
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def _remove_managed_outputs(output_dir: Path) -> None:
    """Remove only files managed by an earlier measurement batch."""
    patterns = (
        "*_frames.csv",
        "*_regions.csv",
        "*_masks.npz",
        "*_internal_structures.gif",
    )
    for pattern in patterns:
        for path in output_dir.rglob(pattern):
            path.unlink()
    for filename in (
        "internal_structure_summary.csv",
        "internal_structure_analysis.json",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()


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


def _validate_input_output_paths(input_path: Path, output_dir: Path) -> None:
    """Prevent generated masks from being rediscovered as checkpoints."""
    resolved_input = input_path.expanduser().resolve()
    input_directory = (
        resolved_input if resolved_input.is_dir() else resolved_input.parent
    )
    resolved_output = output_dir.expanduser().resolve()
    if resolved_output == input_directory or resolved_output.is_relative_to(
        input_directory
    ):
        raise ValueError(
            "Internal-structure output directory must be outside the selected "
            "checkpoint directory."
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
    "light_area_fraction",
    "filament_area_fraction",
    "filament_length_px",
    "bubble_area_fraction",
    "bubble_count",
    "region_count",
    "noise_sigma",
    "status",
    "error",
]

_REGION_FIELDS = [
    "frame_index",
    "region_label",
    "structure_type",
    "polarity",
    "area_px",
    "centroid_y",
    "centroid_x",
    "bbox_min_y",
    "bbox_min_x",
    "bbox_max_y",
    "bbox_max_x",
    "mean_signed_residual",
    "skeleton_length_px",
]

_SUMMARY_FIELDS = [
    "file",
    "source_video",
    "frames",
    "analyzed_frames",
    "extraction_failures",
    "qc_rejected",
    "measurement_failures",
    "median_area_fraction",
    "upper_area_fraction",
    "frame_prevalence",
    "median_light_area_fraction",
    "median_filament_area_fraction",
    "median_filament_length_px",
    "median_bubble_area_fraction",
    "median_bubble_count",
    "status",
    "error",
]
