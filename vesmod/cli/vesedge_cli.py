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
import nd2
import numpy as np

from . import internal_structures_cli
from vesmod.VesEdge import (
    EdgeExtractionConfig,
    EdgeQCConfig,
    QCFlag,
    VesicleEdges,
    VesicleVideo,
)


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
        help="An .nd2 file or a directory containing .nd2 files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when input_path is a directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for checkpoints and GIFs. By default, outputs are "
            "written beside each input file."
        ),
    )
    parser.add_argument(
        "--pixels-per-micron",
        type=float,
        default=1.0,
        help="Pixels per micron in the microscope image. Default: 1.",
    )
    parser.add_argument(
        "--downsample",
        action="store_true",
        help=(
            "Downsample edge-extraction outputs to --n-samples evenly spaced "
            "angular values."
        ),
    )
    parser.add_argument(
        "--n-samples",
        default=120,
        type=int,
        help="Angular samples used with --downsample. Default: 120.",
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
        "--no-gif",
        action="store_true",
        help="Do not save a GIF showing the extracted contours.",
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
        help=(
            "Directory for QC-filtered .npy files, vesedge_qc.json, and "
            "qc_summary.csv. Use a separate directory for each QC configuration."
        ),
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=5.0,
        help=(
            "Maximum wrapped finite second difference allowed by curvature QC. "
            "Default: 5."
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing filtered .npy outputs and QC provenance.",
    )


def iter_input_files(
    input_path: Path,
    suffix: str,
    recursive: bool,
) -> list[Path]:
    """Return selected input files with the requested suffix."""
    input_path = input_path.expanduser().resolve()
    suffix = suffix.lower()
    if input_path.is_file():
        if input_path.suffix.lower() != suffix:
            raise ValueError(f"Expected a {suffix} file, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        candidate
        for candidate in input_path.glob(pattern)
        if candidate.is_file() and candidate.suffix.lower() == suffix
    )


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


def _relative_input_path(path: Path, input_path: Path) -> Path:
    """Return one selected file relative to the user-selected input root."""
    resolved_path = path.expanduser().resolve()
    resolved_input = input_path.expanduser().resolve()
    if resolved_path == resolved_input:
        return Path(resolved_path.name)
    return resolved_path.relative_to(resolved_input)


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
    gif_path = output_base.with_suffix(".gif")

    if checkpoint_path.exists() and not args.overwrite:
        print(f"Skipping {path.expanduser().resolve()}: checkpoint already exists: {checkpoint_path.expanduser().resolve()}")
        return

    if args.extractor_file is not None:
        extractor_func = load_extractor_from_file(
            args.extractor_file,
            args.extractor_name,
        )
    else:
        extractor_func = load_extractor_from_module(args.extractor)

    print(f"Extracting {path.expanduser().resolve()}")
    intensities = nd2.imread(path)
    extraction_config = EdgeExtractionConfig(
        pixels_per_micron=args.pixels_per_micron,
        n_angular_samples=(args.n_samples if args.downsample else None),
    )
    video = VesicleVideo(intensities)
    video.source_path = Path(path)

    try:
        edges = video.extract_edges(extractor_func, extraction_config)
        edges.save_checkpoint(checkpoint_path)
    except (IndexError, ValueError) as error:
        print(f"Failed to extract {path.expanduser().resolve()}: {error}")
        return

    if not args.no_gif and (args.overwrite or not gif_path.exists()):
        video.make_vesicle_gif(gif_path, edges)


def _qc_config_from_args(args: argparse.Namespace) -> EdgeQCConfig:
    """Build the QC configuration requested on the command line."""
    return EdgeQCConfig(
        curvature_threshold=args.curvature_threshold,
        enable_curvature_qc=not args.no_curvature_qc,
        max_relative_area_deviation=args.max_relative_area_deviation,
        enable_area_qc=not args.no_area_qc,
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
    }
    return provenance


def _remove_managed_qc_artifacts(output_dir: Path) -> None:
    """Remove filtered arrays and metadata managed by a previous QC batch."""
    for output_path in output_dir.rglob("*.npy"):
        output_path.unlink()
    for pattern in ("*.area_qc.png", "*.area_qc.csv"):
        for output_path in output_dir.rglob(pattern):
            output_path.unlink()
    for filename in ("qc_summary.csv", "vesedge_qc.json"):
        output_path = output_dir / filename
        if output_path.exists():
            output_path.unlink()


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
        if existing == provenance:
            return
        if not overwrite:
            raise ValueError(
                "QC output directory already contains results from a different "
                "input selection or QC configuration. Choose another --output-dir "
                "or use --overwrite."
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
    accepted = sum(detection.qc.passed for detection in successful)
    return {
        "file": str(_relative_input_path(path, input_path)),
        "frames": len(edges.detections),
        "successful_detections": len(successful),
        "extraction_failures": len(edges.detections) - len(successful),
        "curvature_rejected": curvature_rejected,
        "area_rejected": area_rejected,
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
        "accepted": 0,
        "accepted_fraction": 0.0,
        "status": "load_error",
        "error": error,
    }


def process_qc_file(
    path: Path,
    args: argparse.Namespace,
    qc_config: EdgeQCConfig,
) -> dict:
    """Apply QC to one checkpoint and return its batch summary row."""
    output_path = (
        args.output_dir
        / _relative_input_path(path, args.input_path).with_suffix(".npy")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_exists = output_path.exists()
    if output_exists and not args.overwrite:
        print(f"Keeping existing output for {path.expanduser().resolve()}: {output_path.expanduser().resolve()}")

    try:
        edges = VesicleEdges.from_checkpoint(path)
    except (FileNotFoundError, ValueError) as error:
        message = str(error)
        print(f"Failed to load {path.expanduser().resolve()}: {message}")
        return _load_error_summary(path, args.input_path, message)

    status = "ok"
    qc_error = ""
    try:
        edges.run_qc(qc_config)
    except ValueError as error:
        qc_error = str(error)
        if edges.qc_result is None:
            status = "qc_error"
            print(f"QC failed for {path.expanduser().resolve()}: {error}")
        else:
            status = "no_accepted_frames"
            print(f"QC produced no accepted frames for {path.expanduser().resolve()}: {error}")

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
    has_area_result = (
        edges.qc_result is not None
        and getattr(edges.qc_result, "area", None) is not None
    )
    if has_area_result and (args.overwrite or not area_plot_path.exists()):
        _save_area_qc_plot(area_plot_path, edges)
    if has_area_result and (args.overwrite or not area_csv_path.exists()):
        _write_area_qc_csv(area_csv_path, edges)
    if (
        status == "ok"
        and row["accepted"] > 0
        and (args.overwrite or not output_exists)
    ):
        edges.save_edge_to_npy(output_path)
    return row


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


def _save_area_qc_plot(path: Path, edges: VesicleEdges) -> None:
    """Plot contour area by source frame with configured acceptance bounds."""
    area_result = edges.qc_result.area
    config = edges.qc_result.config
    detections = edges.successful_detections
    frame_indices = [edge.frame_index for edge in detections]
    areas = area_result.areas_pixels2
    reference = area_result.reference_area_pixels2
    deviation = config.max_relative_area_deviation
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
    paths = iter_input_files(args.input_path, ".nd2", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .nd2 files found in {args.input_path}")

    for path in paths:
        process_extract_file(path, args)


def _run_qc(args: argparse.Namespace) -> None:
    """Run one QC configuration over the selected checkpoints."""
    paths = iter_input_files(args.input_path, ".npz", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {args.input_path}")

    qc_config = _qc_config_from_args(args)
    _write_qc_provenance(
        args.output_dir,
        qc_config,
        args.input_path,
        args.recursive,
        paths,
        args.overwrite,
    )
    rows = [
        process_qc_file(path, args, qc_config)
        for path in paths
    ]
    _write_qc_summary(args.output_dir, rows)


def main() -> None:
    """Run the selected VesEdge subcommand."""
    args = parse_args()
    if args.command == "extract":
        _run_extract(args)
    elif args.command == "qc":
        _run_qc(args)
    else:
        internal_structures_cli.run(args)


if __name__ == "__main__":
    main()

