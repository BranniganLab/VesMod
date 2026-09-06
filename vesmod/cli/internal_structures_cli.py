"""CLI orchestration for experimental internal-structure measurements."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import nd2
import numpy as np

from vesmod.VesEdge import (
    EdgeDetection,
    EdgeQCConfig,
    VesicleEdges,
    VesicleVideo,
)
from vesmod.VesEdge.experimental import (
    InternalStructureConfig,
    InternalStructureFrameResult,
    detect_internal_structures,
    summarize_internal_structures,
)

from .path_utils import (
    _display_path,
    _relative_input_path,
    remove_manifest_artifacts,
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
        "--min-light-circularity",
        type=float,
        default=0.2,
        help="Minimum circularity for a bright internal vesicle. Default: 0.2.",
    )
    parser.add_argument(
        "--min-light-solidity",
        type=float,
        default=0.8,
        help="Minimum solidity for a bright internal vesicle. Default: 0.8.",
    )
    parser.add_argument(
        "--max-light-eccentricity",
        type=float,
        default=0.95,
        help="Maximum bright-vesicle eccentricity. Default: 0.95.",
    )
    parser.add_argument(
        "--structure-boundary-exclusion-px",
        type=int,
        default=20,
        help=(
            "Pixels excluded from structure candidates near the detected "
            "outer contour. Default: 20."
        ),
    )
    parser.add_argument(
        "--filament-seed-threshold",
        type=float,
        default=0.7,
        help="High-confidence ridge-evidence seed threshold. Default: 0.7.",
    )
    parser.add_argument(
        "--filament-grow-threshold",
        type=float,
        default=0.35,
        help="Lower dark-or-light ridge threshold for growth. Default: 0.35.",
    )
    parser.add_argument(
        "--filament-scales-px",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
        help="Dark and light curvilinear scales evaluated in pixels.",
    )
    parser.add_argument(
        "--min-filament-length-px",
        type=int,
        default=20,
        help="Minimum skeleton length retained as a filament. Default: 20.",
    )
    parser.add_argument(
        "--bubble-edge-sigma",
        type=float,
        default=2.0,
        help=(
            "Negative-residual threshold for bubble boundaries and compact "
            "dark-region masks; affects dark-region measurements and merged "
            "structure output."
        ),
    )
    parser.add_argument(
        "--bubble-edge-grow-sigma",
        type=float,
        default=1.0,
        help=(
            "Lower negative-residual threshold for growing bubble boundaries "
            "and compact dark-region masks; its absolute magnitude also gates "
            "curvilinear ridge evidence and can affect dark-region "
            "measurements and merged structure output."
        ),
    )
    parser.add_argument(
        "--bubble-closing-px",
        type=int,
        default=4,
        help="Maximum local gap closed in candidate bubble edges.",
    )
    parser.add_argument(
        "--min-bubble-area-px",
        type=int,
        default=100,
        help=(
            "Minimum area for enclosed bubbles and compact dark-region masks; "
            "affects dark-region measurements and merged structure output. "
            "Default: 100."
        ),
    )
    parser.add_argument(
        "--min-bubble-boundary-fraction",
        type=float,
        default=0.45,
        help="Minimum fraction of an enclosed boundary supported by dark pixels.",
    )
    parser.add_argument(
        "--min-bubble-circularity",
        type=float,
        default=0.2,
        help=(
            "Minimum circularity for dark-edged bubbles and compact "
            "dark-region masks; affects dark-region measurements and merged "
            "structure output. Default: 0.2."
        ),
    )
    parser.add_argument(
        "--min-bubble-solidity",
        type=float,
        default=0.8,
        help=(
            "Minimum solidity for dark-edged bubbles and compact dark-region "
            "masks; affects dark-region measurements and merged structure "
            "output. Default: 0.8."
        ),
    )
    parser.add_argument(
        "--max-bubble-eccentricity",
        type=float,
        default=0.95,
        help=(
            "Maximum eccentricity for dark-edged bubbles and compact "
            "dark-region masks; affects dark-region measurements and merged "
            "structure output. Default: 0.95."
        ),
    )
    parser.add_argument(
        "--max-bubble-area-fraction",
        type=float,
        default=0.5,
        help="Largest usable-interior fraction one bubble may occupy. Default: 0.5.",
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
        light_grow_sigma=args.light_grow_sigma,
        min_light_circularity=args.min_light_circularity,
        min_light_solidity=args.min_light_solidity,
        max_light_eccentricity=args.max_light_eccentricity,
        structure_boundary_exclusion_px=args.structure_boundary_exclusion_px,
        filament_seed_threshold=args.filament_seed_threshold,
        filament_grow_threshold=args.filament_grow_threshold,
        filament_scales_px=tuple(args.filament_scales_px),
        min_filament_length_px=args.min_filament_length_px,
        bubble_edge_sigma=args.bubble_edge_sigma,
        bubble_edge_grow_sigma=args.bubble_edge_grow_sigma,
        bubble_closing_px=args.bubble_closing_px,
        min_bubble_area_px=args.min_bubble_area_px,
        min_bubble_boundary_fraction=args.min_bubble_boundary_fraction,
        min_bubble_circularity=args.min_bubble_circularity,
        min_bubble_solidity=args.min_bubble_solidity,
        max_bubble_eccentricity=args.max_bubble_eccentricity,
        max_bubble_area_fraction=args.max_bubble_area_fraction,
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
    video_index = _build_video_filename_index(paths, args.video_root)
    managed_outputs: set[Path] = set()
    summary_rows = [
        process_checkpoint(
            path,
            args,
            config,
            qc_config,
            video_index,
            managed_outputs,
        )
        for path in paths
    ]
    _write_csv(
        args.output_dir / "internal_structure_summary.csv",
        summary_rows,
        _SUMMARY_FIELDS,
    )
    _record_managed_outputs(args.output_dir, managed_outputs)


def process_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    config: InternalStructureConfig,
    qc_config: EdgeQCConfig | None,
    video_index: dict[str, tuple[Path, ...]] | None = None,
    managed_outputs: set[Path] | None = None,
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
            video_index,
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
            _apply_qc(edges, qc_config, frames)
    except (OSError, IndexError, TypeError, ValueError) as error:
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
        if qc_config is not None and (
            not edges.qc_result.passed or not edge_result.qc.passed
        ):
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

    frames_path = output_base.with_name(output_base.name + "_frames.csv")
    _write_csv(
        frames_path,
        frame_rows,
        _FRAME_FIELDS,
    )
    if managed_outputs is not None:
        managed_outputs.add(frames_path)
    regions_path = output_base.with_name(output_base.name + "_regions.csv")
    _write_csv(
        regions_path,
        region_rows,
        _REGION_FIELDS,
    )
    if managed_outputs is not None:
        managed_outputs.add(regions_path)
    if args.save_masks:
        _save_masks(output_base, results, frames.shape[1:])
        if managed_outputs is not None:
            managed_outputs.add(
                output_base.with_name(output_base.name + "_masks.npz")
            )
    if not args.no_gif:
        _save_overlay_gif(output_base, frames, edges, results)
        if managed_outputs is not None:
            managed_outputs.add(
                output_base.with_name(
                    output_base.name + "_internal_structures.gif"
                )
            )

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
        qc_config = EdgeQCConfig.from_dict(provenance["qc_config"])
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


def _apply_qc(
    edges: VesicleEdges,
    qc_config: EdgeQCConfig,
    frames: np.ndarray,
) -> None:
    """Apply frame eligibility while allowing a result with zero passing frames."""
    try:
        edges.run_qc(qc_config, frames)
    except ValueError:
        if edges.qc_result is None:
            raise


def _build_video_filename_index(
    checkpoint_paths: list[Path],
    video_root: Path | None,
) -> dict[str, tuple[Path, ...]]:
    """Index video filenames once for the selected checkpoint batch."""
    search_roots = {
        path.expanduser().resolve().parent
        for path in checkpoint_paths
    }
    if video_root is not None:
        resolved_root = video_root.expanduser().resolve()
        if resolved_root.is_dir():
            search_roots.add(resolved_root)

    index: dict[str, set[Path]] = {}
    for root in search_roots:
        for candidate in root.rglob("*"):
            if candidate.is_file():
                index.setdefault(candidate.name.lower(), set()).add(
                    candidate.resolve()
                )
    return {
        filename: tuple(sorted(paths))
        for filename, paths in index.items()
    }


def _resolve_video_path(
    stored_path: str | Path | None,
    video_root: Path | None,
    checkpoint_path: Path,
    video_index: dict[str, tuple[Path, ...]] | None = None,
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

    matches = _find_video_matches(video_name, search_roots, video_index)
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
    video_index: dict[str, tuple[Path, ...]] | None = None,
) -> list[Path]:
    """Find unique case-insensitive filename matches below selected roots."""
    lowercase_name = video_name.lower()
    if video_index is not None:
        return sorted(
            candidate
            for candidate in video_index.get(lowercase_name, ())
            if any(candidate.is_relative_to(root) for root in search_roots)
        )

    matches: set[Path] = set()
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
        "structure_count": result.structure_count,
        "light_area_fraction": result.light_area_fraction,
        "dark_region_area_fraction": result.dark_region_area_fraction,
        "filament_area_fraction": result.filament_area_fraction,
        "filament_length_px": result.filament_length_px,
        "bubble_area_fraction": result.bubble_area_fraction,
        "bubble_count": result.bubble_count,
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
        "structure_count": "",
        "light_area_fraction": "",
        "dark_region_area_fraction": "",
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
                "evidence_types": ";".join(region.evidence_types),
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
            "median_dark_region_area_fraction": "",
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
        "median_dark_region_area_fraction": (
            summary.median_dark_region_area_fraction
        ),
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
        "median_dark_region_area_fraction": "",
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
                _to_full_frame_channel_mask(results[index], "light_region")
                for index in frame_indices
            ]
        )
        filament_masks = np.stack(
            [
                _to_full_frame_channel_mask(results[index], "dark_filament")
                for index in frame_indices
            ]
        )
        bubble_masks = np.stack(
            [
                _to_full_frame_channel_mask(results[index], "bubble")
                for index in frame_indices
            ]
        )
        dark_region_masks = np.stack(
            [
                _to_full_frame_channel_mask(results[index], "dark_region")
                for index in frame_indices
            ]
        )
    else:
        masks = np.empty((0, *frame_shape), dtype=bool)
        light_masks = masks.copy()
        filament_masks = masks.copy()
        bubble_masks = masks.copy()
        dark_region_masks = masks.copy()
    path = output_base.with_name(output_base.name + "_masks.npz")
    np.savez_compressed(
        path,
        frame_indices=frame_indices,
        structure_masks=masks,
        light_region_masks=light_masks,
        dark_filament_masks=filament_masks,
        bubble_region_masks=bubble_masks,
        dark_region_masks=dark_region_masks,
    )


def _to_full_frame_channel_mask(
    result: InternalStructureFrameResult,
    structure_type: str,
) -> np.ndarray:
    """Map one named structure channel into original-image coordinates."""
    return result.to_full_frame_channel_mask(structure_type)


def _save_overlay_gif(
    output_base: Path,
    frames: np.ndarray,
    edges: VesicleEdges,
    results: dict[int, InternalStructureFrameResult],
) -> None:
    """Save one merged structure overlay through the shared GIF renderer."""
    def add_structure_overlays(axis, frame_index: int) -> None:
        result = results.get(frame_index)
        if result is None:
            return
        full_mask = result.to_full_frame_mask()
        overlay = np.ma.masked_where(~full_mask, full_mask)
        axis.imshow(overlay, cmap="winter", alpha=0.55, vmin=0, vmax=1)

    def structure_title(frame_index: int) -> str:
        result = results.get(frame_index)
        if result is None:
            return f"frame {frame_index}: not analyzed"
        return (
            f"frame {frame_index}: "
            f"structured={result.structured_area_fraction:.3f}, "
            f"regions={result.structure_count}"
        )

    path = output_base.with_name(output_base.name + "_internal_structures.gif")
    VesicleVideo(frames, source_path=edges.source_path).make_vesicle_gif(
        path,
        edges,
        frame_decorator=add_structure_overlays,
        title_provider=structure_title,
    )


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
        "managed_artifacts": [],
    }
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        comparable = dict(existing) if isinstance(existing, dict) else {}
        comparable["managed_artifacts"] = []
        if comparable != provenance:
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
    provenance_path = output_dir / "internal_structure_analysis.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            "Existing internal-structure provenance is malformed; refusing "
            "to remove files."
        ) from error
    if (
        not isinstance(provenance, dict)
        or provenance.get("experimental_method") != "internal_structures"
    ):
        raise ValueError(
            "Existing internal-structure provenance is incomplete; refusing "
            "to remove files."
        )
    remove_manifest_artifacts(
        output_dir,
        provenance,
        manifest_key="managed_artifacts",
        manifest_name="internal-structure provenance",
        allowed_suffixes={".csv", ".npz", ".gif"},
        metadata_files=(
            "internal_structure_summary.csv",
            "internal_structure_analysis.json",
        ),
    )


def _record_managed_outputs(
    output_dir: Path,
    managed_outputs: set[Path],
) -> None:
    """Record files created for each checkpoint in the batch manifest."""
    provenance_path = output_dir / "internal_structure_analysis.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["managed_artifacts"] = sorted(
        str(path.resolve().relative_to(output_dir.resolve()))
        for path in managed_outputs
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


_FRAME_FIELDS = [
    "frame_index",
    "usable_area_px",
    "structured_area_px",
    "structured_area_fraction",
    "structure_count",
    "light_area_fraction",
    "dark_region_area_fraction",
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
    "evidence_types",
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
    "median_dark_region_area_fraction",
    "median_filament_area_fraction",
    "median_filament_length_px",
    "median_bubble_area_fraction",
    "median_bubble_count",
    "status",
    "error",
]
