"""Command-line interface for VesEdge extraction and independent analyses."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from . import internal_structures_batch_cli as internal_structures_cli
from .gif_cli import add_gif_parser, run_gif
from .input_selection import InputPathsAction, select_input_files
from .path_utils import (
    _display_path,
    _relative_input_path,
    remove_manifest_artifacts,
)
from vesmod.VesEdge import (
    AreaQCConfig,
    CurvatureQCConfig,
    EdgeExtractionConfig,
    EdgeQCConfig,
    QCFlag,
    TrajectoryQCFlag,
    VesicleEdges,
    VesicleVideo,
    open_frame_source,
)
from vesmod.VesEdge.experimental import InternalVesicleQCConfig


def _parse_angular_samples(value: str) -> int | None:
    """Parse an angular sample count or the explicit ``native`` sentinel."""
    if value.lower() == "native":
        return None
    try:
        samples = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "n-angular-samples must be a positive integer or 'native'."
        ) from error
    if samples <= 0:
        raise argparse.ArgumentTypeError(
            "n-angular-samples must be a positive integer or 'native'."
        )
    return samples


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract vesicle edges to reusable checkpoints, then apply quality "
            "control in a separate stage."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_extract_parser(subparsers)
    _add_qc_parser(subparsers)
    internal_structures_cli.add_parser(subparsers)
    add_gif_parser(subparsers)
    return parser.parse_args()


def _add_extract_parser(subparsers) -> None:
    """Add arguments for the extraction stage."""
    parser = subparsers.add_parser(
        "extract",
        help="Extract edges from .nd2 files and save .npz checkpoints.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="+",
        action=InputPathsAction,
        help="One or more .nd2 files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for directory and glob inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for checkpoints. By default, outputs are written "
            "beside each input file."
        ),
    )
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument(
        "--pixels-per-micron",
        type=float,
        help="Measured pixels per micron in the microscope image.",
    )
    calibration.add_argument(
        "--assume-one-pixel-per-micron",
        action="store_true",
        help=(
            "Explicitly use unit calibration (1 pixel per micron). This is "
            "intended for dimensionless/test workflows, not calibrated microscopy."
        ),
    )
    parser.add_argument(
        "--n-angular-samples",
        default=120,
        type=_parse_angular_samples,
        metavar="N|native",
        help=(
            "Analysis-contour angular sampling. Use a positive integer to "
            "resample uniformly or 'native' to retain extractor sampling. "
            "Default: 120."
        ),
    )
    parser.add_argument(
        "--extractor",
        default="vesmod.VesEdge:extract_edge_from_frame",
        help=(
            "Edge extractor as 'module:function'. The function must accept one "
            "2D frame and return (r_vals, vesicle_center)."
        ),
    )
    parser.add_argument(
        "--extractor-file",
        default=None,
        type=Path,
        help="Path to a Python file containing a custom edge extractor.",
    )
    parser.add_argument(
        "--extractor-name",
        default="extract_edge_from_frame",
        help="Name of the extractor function in --extractor-file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extraction outputs.",
    )


def _add_qc_parser(subparsers) -> None:
    """Add arguments for the QC stage."""
    parser = subparsers.add_parser(
        "qc",
        help="Apply QC to .npz checkpoints and save filtered .npy files.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="+",
        action=InputPathsAction,
        help="One or more VesEdge .npz files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for directory and glob inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for QC-filtered .npy files, vesedge_qc.json, and "
            "qc_summary.csv. Use a separate directory for each QC configuration."
        ),
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=0.059,
        help=(
            "Maximum dimensionless wrapped finite second difference of the "
            "median-radius-normalized contour allowed by curvature QC. "
            "Default: 0.059."
        ),
    )
    parser.add_argument(
        "--no-curvature-qc",
        action="store_true",
        help="Disable frame-level curvature QC.",
    )
    parser.add_argument(
        "--max-relative-area-deviation",
        type=float,
        default=0.25,
        help=(
            "Maximum absolute fractional deviation from the trajectory median "
            "contour area. Default: 0.25."
        ),
    )
    parser.add_argument(
        "--no-area-qc",
        action="store_true",
        help="Disable trajectory-level contour-area deviation QC.",
    )
    parser.add_argument(
        "--internal-vesicle-qc",
        action="store_true",
        help=(
            "Check whether the edge detector persistently traced a smaller "
            "vesicle enclosed by the intended vesicle. Disabled by default."
        ),
    )
    parser.add_argument(
        "--max-internal-vesicle-area-fraction",
        type=float,
        default=0.5,
        help=(
            "Inspect for a larger enclosing membrane only when the median "
            "traced contour occupies less than this fraction of the image. "
            "Default: 0.5."
        ),
    )
    parser.add_argument(
        "--internal-vesicle-min-frame-fraction",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of inspected frames with enclosing-boundary "
            "evidence required to reject the video. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--internal-vesicle-max-frames",
        type=int,
        default=20,
        help=(
            "Maximum number of frames sampled evenly across the video for "
            "internal-vesicle QC. Default: 20."
        ),
    )
    parser.add_argument(
        "--internal-vesicle-min-valid-frames",
        type=int,
        default=3,
        help=(
            "Minimum valid sampled frames required for a decision, capped by "
            "the number sampled for short videos. Default: 3."
        ),
    )
    parser.add_argument(
        "--internal-vesicle-min-valid-frame-fraction",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of sampled frames that must yield valid scores. "
            "Default: 0.5."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing filtered .npy outputs and QC provenance.",
    )


def iter_input_files(
    input_path: Path | list[Path],
    suffix: str,
    recursive: bool,
) -> list[Path]:
    """Return selected input files with the requested suffix."""
    paths, _ = select_input_files(input_path, suffix, recursive)
    return paths


def load_extractor_from_module(import_string: str):
    """Load an edge extractor from 'module:function' syntax."""
    module_name, func_name = import_string.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    extractor = getattr(module, func_name)
    if not callable(extractor):
        raise TypeError(f"{import_string} is not callable.")
    return extractor


def load_extractor_from_file(file_path: Path, function_name: str):
    """Load an edge extractor function from a Python file."""
    file_path = file_path.expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Extractor file does not exist: {file_path}")

    spec = importlib.util.spec_from_file_location(
        "custom_vesedge_extractor",
        file_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import extractor file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extractor = getattr(module, function_name)
    if not callable(extractor):
        raise TypeError(f"{function_name} in {file_path} is not callable.")
    return extractor


def _output_base(
    path: Path,
    input_path: Path,
    output_dir: Path | None,
) -> Path:
    """Return an output path stem while preserving relative input directories."""
    if output_dir is None:
        return path.with_suffix("")
    output_base = output_dir / _relative_input_path(path, input_path).with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    return output_base


def process_extract_file(path: Path, args: argparse.Namespace) -> None:
    """Extract one ND2 video and save a reusable checkpoint."""
    output_base = _output_base(path, args.input_path, args.output_dir)
    checkpoint_path = output_base.with_suffix(".npz")
    if checkpoint_path.exists() and not args.overwrite:
        print(
            f"Skipping {_display_path(path)}: checkpoint already exists: "
            f"{_display_path(checkpoint_path)}"
        )
        return

    if args.extractor_file is not None:
        extractor_func = load_extractor_from_file(
            args.extractor_file,
            args.extractor_name,
        )
    else:
        extractor_func = load_extractor_from_module(args.extractor)

    print(f"Extracting {_display_path(path)}")
    assumed_unit_calibration = getattr(
        args,
        "assume_one_pixel_per_micron",
        False,
    )
    pixels_per_micron = (
        1.0 if assumed_unit_calibration else args.pixels_per_micron
    )
    if hasattr(args, "n_angular_samples"):
        n_angular_samples = args.n_angular_samples
    else:
        n_angular_samples = args.n_samples if args.downsample else None
    extraction_config = EdgeExtractionConfig(
        pixels_per_micron=pixels_per_micron,
        n_angular_samples=n_angular_samples,
        calibration_source=(
            "assumed"
            if assumed_unit_calibration
            else (
                "measured"
                if hasattr(args, "assume_one_pixel_per_micron")
                else "unspecified"
            )
        ),
    )
    try:
        with open_frame_source(path) as frames:
            video = VesicleVideo(frames)
            video.source_path = Path(path)
            edges = video.extract_edges(extractor_func, extraction_config)
            edges.save_checkpoint(checkpoint_path)
    except (IndexError, ValueError) as error:
        print(f"Failed to extract {_display_path(path)}: {error}")
        return


def _qc_config_from_args(args: argparse.Namespace) -> EdgeQCConfig:
    """Build the QC configuration requested on the command line."""
    return EdgeQCConfig(
        curvature=CurvatureQCConfig(
            threshold=args.curvature_threshold,
            enabled=not args.no_curvature_qc,
        ),
        area=AreaQCConfig(
            max_relative_deviation=args.max_relative_area_deviation,
            enabled=not args.no_area_qc,
        ),
        internal_vesicle=InternalVesicleQCConfig(
            enabled=getattr(args, "internal_vesicle_qc", False),
            max_area_fraction=getattr(
                args, "max_internal_vesicle_area_fraction", 0.5
            ),
            min_frame_fraction=getattr(
                args, "internal_vesicle_min_frame_fraction", 0.5
            ),
            max_frames=getattr(args, "internal_vesicle_max_frames", 20),
            min_valid_frames=getattr(
                args, "internal_vesicle_min_valid_frames", 3
            ),
            min_valid_frame_fraction=getattr(
                args, "internal_vesicle_min_valid_frame_fraction", 0.5
            ),
        ),
    )


def _qc_provenance(
    qc_config: EdgeQCConfig,
    input_path: Path,
    recursive: bool,
    paths: list[Path],
) -> dict:
    """Return serializable provenance for one resolved QC batch."""
    provenance = {
        "input_path": str(input_path.expanduser().resolve()),
        "recursive": recursive,
        "checkpoint_manifest": [str(path.resolve()) for path in paths],
        "qc_config": asdict(qc_config),
        "managed_artifacts": [],
    }
    return provenance


def _remove_managed_qc_artifacts(output_dir: Path) -> None:
    """Remove filtered arrays and metadata managed by a previous QC batch."""
    provenance_path = output_dir / "vesedge_qc.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            "Existing VesEdge QC provenance is malformed; refusing to remove "
            "files."
        ) from error
    if not isinstance(provenance, dict) or not {
        "checkpoint_manifest",
        "qc_config",
    }.issubset(provenance):
        raise ValueError(
            "Existing VesEdge QC provenance is incomplete; refusing to remove "
            "files."
        )
    remove_manifest_artifacts(
        output_dir,
        provenance,
        manifest_key="managed_artifacts",
        manifest_name="VesEdge QC provenance",
        allowed_suffixes={".npy", ".png", ".csv"},
        metadata_files=("qc_summary.csv", "vesedge_qc.json"),
    )


def _write_qc_provenance(
    output_dir: Path,
    qc_config: EdgeQCConfig,
    input_path: Path,
    recursive: bool,
    paths: list[Path],
    overwrite: bool,
) -> None:
    """Write QC provenance and reject incompatible existing provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "vesedge_qc.json"
    provenance = _qc_provenance(
        qc_config,
        input_path,
        recursive,
        paths,
    )

    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        comparable = dict(existing) if isinstance(existing, dict) else {}
        comparable["managed_artifacts"] = []
        if comparable == provenance:
            if overwrite:
                _remove_managed_qc_artifacts(output_dir)
                # The matching-provenance cleanup already removed the old
                # manifest; continue by writing the replacement below.
            else:
                provenance["managed_artifacts"] = existing.get(
                    "managed_artifacts", []
                )
                provenance_path.write_text(
                    json.dumps(provenance, indent=2) + "\n",
                    encoding="utf-8",
                )
                return
        else:
            if not overwrite:
                raise ValueError(
                    "QC output directory already contains results from a different "
                    "input selection or QC configuration. Choose another "
                    "--output-dir or use --overwrite."
                )
            _remove_managed_qc_artifacts(output_dir)

    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def _qc_summary(
    path: Path,
    input_path: Path,
    edges: VesicleEdges,
    status: str,
    error: str = "",
) -> dict:
    """Build a summary row for one QCed checkpoint."""
    successful = edges.successful_detections
    curvature_rejected = sum(
        QCFlag.CURVATURE in detection.qc.flags
        for detection in successful
    )
    area_rejected = sum(
        QCFlag.AREA_DEVIATION in detection.qc.flags
        for detection in successful
    )
    trajectory_flags = getattr(edges.qc_result, "trajectory_flags", frozenset())
    trajectory_rejected = bool(trajectory_flags)
    accepted = (
        0
        if trajectory_rejected
        else sum(detection.qc.passed for detection in successful)
    )
    internal_result = getattr(
        edges.qc_result,
        "internal_vesicle",
        None,
    )
    return {
        "file": str(_relative_input_path(path, input_path)),
        "frames": len(edges.detections),
        "successful_detections": len(successful),
        "extraction_failures": len(edges.detections) - len(successful),
        "curvature_rejected": curvature_rejected,
        "area_rejected": area_rejected,
        "internal_vesicle_trajectory_rejected": (
            TrajectoryQCFlag.INTERNAL_VESICLE in trajectory_flags
        ),
        "internal_vesicle_inspected": (
            internal_result.inspected if internal_result is not None else False
        ),
        "internal_vesicle_area_fraction": (
            internal_result.contour_area_fraction
            if internal_result is not None
            else ""
        ),
        "internal_vesicle_positive_frame_fraction": (
            internal_result.positive_frame_fraction
            if internal_result is not None
            else ""
        ),
        "internal_vesicle_valid_frame_count": (
            internal_result.valid_frame_count
            if internal_result is not None
            else ""
        ),
        "internal_vesicle_valid_frame_fraction": (
            internal_result.valid_frame_fraction
            if internal_result is not None
            else ""
        ),
        "internal_vesicle_reason": (
            internal_result.reason if internal_result is not None else ""
        ),
        "accepted": accepted,
        "accepted_fraction": accepted / len(successful),
        "status": status,
        "error": error,
    }


def _load_error_summary(path: Path, input_path: Path, error: str) -> dict:
    """Build a canonical summary row for a checkpoint that could not load."""
    return {
        "file": str(_relative_input_path(path, input_path)),
        "frames": 0,
        "successful_detections": 0,
        "extraction_failures": 0,
        "curvature_rejected": 0,
        "area_rejected": 0,
        "internal_vesicle_trajectory_rejected": False,
        "internal_vesicle_inspected": False,
        "internal_vesicle_area_fraction": "",
        "internal_vesicle_positive_frame_fraction": "",
        "internal_vesicle_valid_frame_count": "",
        "internal_vesicle_valid_frame_fraction": "",
        "internal_vesicle_reason": "",
        "accepted": 0,
        "accepted_fraction": 0.0,
        "status": "load_error",
        "error": error,
    }


def process_qc_file(
    path: Path,
    args: argparse.Namespace,
    qc_config: EdgeQCConfig,
    managed_artifacts: set[Path] | None = None,
) -> dict:
    """Apply QC to one checkpoint and return its batch summary row."""
    output_path = (
        args.output_dir
        / _relative_input_path(path, args.input_path).with_suffix(".npy")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_exists = output_path.exists()
    if output_exists and not args.overwrite:
        print(
            f"Keeping existing output for {_display_path(path)}: "
            f"{_display_path(output_path)}"
        )

    try:
        edges = VesicleEdges.from_checkpoint(path)
    except (FileNotFoundError, ValueError) as error:
        message = str(error)
        print(f"Failed to load {_display_path(path)}: {message}")
        return _load_error_summary(path, args.input_path, message)

    status = "ok"
    qc_error = ""
    try:
        frames = None
        if qc_config.internal_vesicle.enabled:
            if edges.source_path is None:
                raise ValueError(
                    "Internal-vesicle QC requires a checkpoint with a source "
                    "video path."
                )
            frames = open_frame_source(edges.source_path)
        if frames is None:
            edges.run_qc(qc_config)
        else:
            with frames:
                edges.run_qc(qc_config, frames=frames)
    except (OSError, ValueError) as error:
        qc_error = str(error)
        if edges.qc_result is None:
            status = "qc_error"
            print(f"QC failed for {_display_path(path)}: {error}")
        else:
            status = "no_accepted_frames"
            print(
                "QC produced no accepted frames for "
                f"{_display_path(path)}: {error}"
            )

    if status == "no_accepted_frames" and args.overwrite and output_exists:
        output_path.unlink()

    row = _qc_summary(
        path,
        args.input_path,
        edges,
        status,
        qc_error,
    )
    area_plot_path = output_path.with_suffix(".area_qc.png")
    area_csv_path = output_path.with_suffix(".area_qc.csv")
    internal_vesicle_csv_path = output_path.with_suffix(
        ".internal_vesicle_qc.csv"
    )
    has_area_result = (
        edges.qc_result is not None
        and getattr(edges.qc_result, "area", None) is not None
    )
    if has_area_result and (args.overwrite or not area_plot_path.exists()):
        _save_area_qc_plot(area_plot_path, edges)
        if managed_artifacts is not None:
            managed_artifacts.add(area_plot_path)
    if has_area_result and (args.overwrite or not area_csv_path.exists()):
        _write_area_qc_csv(area_csv_path, edges)
        if managed_artifacts is not None:
            managed_artifacts.add(area_csv_path)
    has_internal_vesicle_result = (
        edges.qc_result is not None
        and getattr(edges.qc_result, "internal_vesicle", None) is not None
    )
    if has_internal_vesicle_result and (
        args.overwrite or not internal_vesicle_csv_path.exists()
    ):
        _write_internal_vesicle_qc_csv(internal_vesicle_csv_path, edges)
        if managed_artifacts is not None:
            managed_artifacts.add(internal_vesicle_csv_path)
    if (
        status == "ok"
        and row["accepted"] > 0
        and (args.overwrite or not output_exists)
    ):
        edges.save_edge_to_npy(output_path)
        if managed_artifacts is not None:
            managed_artifacts.add(output_path)
    return row


def _record_qc_artifacts(
    output_dir: Path,
    managed_artifacts: set[Path],
) -> None:
    """Record the filtered arrays and diagnostics created by this batch."""
    provenance_path = output_dir / "vesedge_qc.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["managed_artifacts"] = sorted(
        str(path.resolve().relative_to(output_dir.resolve()))
        for path in managed_artifacts
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_area_qc_csv(path: Path, edges: VesicleEdges) -> None:
    """Write exact per-frame contour-area QC measurements."""
    area_result = edges.qc_result.area
    rows = []
    for detection, area, deviation in zip(
        edges.successful_detections,
        area_result.areas_pixels2,
        area_result.relative_deviations,
        strict=True,
    ):
        rows.append(
            {
                "frame_index": detection.frame_index,
                "area_pixels2": area,
                "relative_area_deviation": deviation,
                "area_rejected": (
                    QCFlag.AREA_DEVIATION in detection.qc.flags
                ),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "frame_index",
                "area_pixels2",
                "relative_area_deviation",
                "area_rejected",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_internal_vesicle_qc_csv(
    path: Path,
    edges: VesicleEdges,
) -> None:
    """Write per-frame evidence of a wrongly traced internal vesicle."""
    result = edges.qc_result.internal_vesicle
    rows = []
    if result.inspected:
        trajectory_rejected = result.persistent_enclosing_boundary
        for frame_index, score in zip(
            result.sampled_frame_indices,
            result.scores,
            strict=True,
        ):
            rows.append(
                {
                    "frame_index": frame_index,
                    "enclosing_boundary_angular_coverage": score,
                    "internal_vesicle_trajectory_rejected": (
                        trajectory_rejected
                    ),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "frame_index",
                "enclosing_boundary_angular_coverage",
                "internal_vesicle_trajectory_rejected",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_area_qc_plot(path: Path, edges: VesicleEdges) -> None:
    """Plot contour area by source frame with configured acceptance bounds."""
    area_result = edges.qc_result.area
    config = edges.qc_result.config
    detections = edges.successful_detections
    frame_indices = [edge.frame_index for edge in detections]
    areas = area_result.areas_pixels2
    reference = area_result.reference_area_pixels2
    deviation = config.area.max_relative_deviation
    lower_bound = reference * (1 - deviation)
    upper_bound = reference * (1 + deviation)

    figure, axis = plt.subplots()
    axis.plot(frame_indices, areas, ".", color="tab:blue", label="contour area")
    axis.axhline(reference, color="black", label="trajectory median")
    axis.axhline(
        lower_bound,
        color="tab:red",
        linestyle="--",
        label="acceptance bounds",
    )
    axis.axhline(upper_bound, color="tab:red", linestyle="--")
    axis.set_xlabel("Source frame")
    axis.set_ylabel("Contour area (pixels squared)")
    axis.legend()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_qc_summary(output_dir: Path, rows: list[dict]) -> None:
    """Write one CSV row per processed checkpoint."""
    summary_path = output_dir / "qc_summary.csv"
    fieldnames = [
        "file",
        "frames",
        "successful_detections",
        "extraction_failures",
        "curvature_rejected",
        "area_rejected",
        "internal_vesicle_trajectory_rejected",
        "internal_vesicle_inspected",
        "internal_vesicle_area_fraction",
        "internal_vesicle_positive_frame_fraction",
        "internal_vesicle_valid_frame_count",
        "internal_vesicle_valid_frame_fraction",
        "internal_vesicle_reason",
        "accepted",
        "accepted_fraction",
        "status",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_extract(args: argparse.Namespace) -> None:
    """Run extraction over the selected ND2 files."""
    paths, input_root = select_input_files(args.input_path, ".nd2", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .nd2 files found for {args.input_path}")
    args.input_path = input_root

    for path in paths:
        process_extract_file(path, args)


def _run_qc(args: argparse.Namespace) -> None:
    """Run one QC configuration over the selected checkpoints."""
    paths, input_root = select_input_files(args.input_path, ".npz", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npz files found for {args.input_path}")
    args.input_path = input_root

    qc_config = _qc_config_from_args(args)
    _write_qc_provenance(
        args.output_dir,
        qc_config,
        args.input_path,
        args.recursive,
        paths,
        args.overwrite,
    )
    managed_artifacts: set[Path] = set()
    rows = [
        process_qc_file(path, args, qc_config, managed_artifacts)
        for path in paths
    ]
    _write_qc_summary(args.output_dir, rows)
    _record_qc_artifacts(args.output_dir, managed_artifacts)


def main() -> None:
    """Run the selected VesEdge subcommand."""
    args = parse_args()
    if args.command == "extract":
        _run_extract(args)
    elif args.command == "qc":
        _run_qc(args)
    elif args.command == "gif":
        run_gif(args)
    elif args.command == "internal-structures":
        internal_structures_cli.run(args)
    else:
        raise ValueError(f"Unknown VesEdge command: {args.command}")


if __name__ == "__main__":
    main()
