"""Command-line interface for VesEdge extraction and quality control."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import nd2

from vesmod.VesEdge import (
    EdgeExtractionConfig,
    EdgeQCConfig,
    QCFlag,
    VesicleEdges,
    VesicleVideo,
)
from vesmod.VesEdge.population_plotting import save_population_histograms


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
        "--population-bic-threshold",
        type=float,
        default=10.0,
        help=(
            "Minimum BIC improvement required to prefer two center/radius "
            "populations. Default: 10."
        ),
    )
    parser.add_argument(
        "--max-minor-population-fraction",
        type=float,
        default=0.25,
        help=(
            "Maximum fraction assigned to a minor population for automatic "
            "rejection. Default: 0.25."
        ),
    )
    parser.add_argument(
        "--no-curvature-qc",
        action="store_true",
        help="Disable frame-level curvature QC.",
    )
    parser.add_argument(
        "--no-population-qc",
        action="store_true",
        help="Disable trajectory-level center/radius population QC.",
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
        print(f"Skipping {path.name}: checkpoint already exists: {checkpoint_path.name}")
        return

    if args.extractor_file is not None:
        extractor_func = load_extractor_from_file(
            args.extractor_file,
            args.extractor_name,
        )
    else:
        extractor_func = load_extractor_from_module(args.extractor)

    print(f"Extracting {path.name}")
    intensities = nd2.imread(path)
    extraction_config = EdgeExtractionConfig(
        pixels_per_micron=args.pixels_per_micron,
        n_angular_samples=(args.n_samples if args.downsample else None),
    )
    video = VesicleVideo(intensities)

    try:
        edges = video.extract_edges(extractor_func, extraction_config)
        edges.save_checkpoint(checkpoint_path)
    except (IndexError, ValueError) as error:
        print(f"Failed to extract {path.name}: {error}")
        return

    if not args.no_gif and (args.overwrite or not gif_path.exists()):
        video.make_vesicle_gif(gif_path, edges)


def _qc_config_from_args(args: argparse.Namespace) -> EdgeQCConfig:
    """Build the QC configuration requested on the command line."""
    return EdgeQCConfig(
        curvature_threshold=args.curvature_threshold,
        population_bic_threshold=args.population_bic_threshold,
        max_minor_population_fraction=args.max_minor_population_fraction,
        enable_curvature_qc=not args.no_curvature_qc,
        enable_population_qc=not args.no_population_qc,
    )


def _qc_provenance(
    qc_config: EdgeQCConfig,
    input_path: Path,
    recursive: bool,
    paths: list[Path],
) -> dict:
    """Return serializable provenance for one resolved QC batch."""
    return {
        "input_path": str(input_path.expanduser().resolve()),
        "recursive": recursive,
        "checkpoint_manifest": [str(path.resolve()) for path in paths],
        "qc_config": asdict(qc_config),
    }


def _remove_managed_qc_artifacts(output_dir: Path) -> None:
    """Remove filtered arrays and metadata managed by a previous QC batch."""
    for pattern in ("*.npy", "*.population_histograms.png"):
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
    provenance = _qc_provenance(qc_config, input_path, recursive, paths)

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
    population_rejected = sum(
        QCFlag.POPULATION_OUTLIER in detection.qc.flags
        for detection in successful
    )
    accepted = sum(detection.qc.passed for detection in successful)
    return {
        "file": str(_relative_input_path(path, input_path)),
        "frames": len(edges.detections),
        "successful_detections": len(successful),
        "extraction_failures": len(edges.detections) - len(successful),
        "curvature_rejected": curvature_rejected,
        "population_rejected": population_rejected,
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
        "population_rejected": 0,
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
        print(f"Keeping existing output for {path.name}: {output_path.name}")

    try:
        edges = VesicleEdges.from_checkpoint(path)
    except (FileNotFoundError, ValueError) as error:
        message = str(error)
        print(f"Failed to load {path.name}: {message}")
        return _load_error_summary(path, args.input_path, message)

    status = "ok"
    qc_error = ""
    try:
        edges.run_qc(qc_config)
    except ValueError as error:
        qc_error = str(error)
        if edges.qc_result is None:
            status = "qc_error"
            print(f"QC failed for {path.name}: {error}")
        else:
            status = "no_accepted_frames"
            print(f"QC produced no accepted frames for {path.name}: {error}")

    population_figure_path = output_path.with_suffix(
        ".population_histograms.png"
    )
    if (
        edges.qc_result is not None
        and getattr(edges.qc_result, "population", None) is not None
        and (args.overwrite or not population_figure_path.exists())
    ):
        save_population_histograms(
            edges.detections,
            population_figure_path,
        )

    row = _qc_summary(path, args.input_path, edges, status, qc_error)
    if (
        status == "ok"
        and row["accepted"] > 0
        and (args.overwrite or not output_exists)
    ):
        edges.save_edge_to_npy(output_path)
    return row


def _write_qc_summary(output_dir: Path, rows: list[dict]) -> None:
    """Write one CSV row per selected checkpoint in the QC batch."""
    if not rows:
        return
    summary_path = output_dir / "qc_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_extract(args: argparse.Namespace) -> None:
    """Run the extraction subcommand."""
    paths = iter_input_files(args.input_path, ".nd2", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .nd2 files found in {args.input_path}")
    for path in paths:
        process_extract_file(path, args)


def _run_qc(args: argparse.Namespace) -> None:
    """Run the QC subcommand."""
    paths = iter_input_files(args.input_path, ".npz", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {args.input_path}")

    qc_config = _qc_config_from_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    _write_qc_provenance(
        args.output_dir,
        qc_config,
        args.input_path,
        args.recursive,
        paths,
        args.overwrite,
    )

    rows = [process_qc_file(path, args, qc_config) for path in paths]
    _write_qc_summary(args.output_dir, rows)


def main() -> None:
    """Run the VesEdge command-line interface."""
    args = parse_args()
    if args.command == "extract":
        _run_extract(args)
    else:
        _run_qc(args)


if __name__ == "__main__":
    main()
