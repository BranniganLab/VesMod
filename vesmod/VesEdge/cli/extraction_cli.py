"""Command-line extraction of VesEdge checkpoints from source videos."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path

from vesmod.VesEdge.extraction.config import EdgeExtractionConfig
from vesmod.VesEdge.vesicle_video import VesicleVideo
from vesmod.VesEdge.io.frame_source import open_frame_source
from vesmod.io.path_mapping import map_output_path
from vesmod.cli.batch_policy import (
    add_batch_policy_argument,
    exit_code,
    report_batch_summary,
)
from vesmod.cli.input_selection import InputPathsAction, select_input_files
from vesmod.cli.path_utils import _display_path


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


def add_parser(subparsers) -> None:
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
        default="vesmod.VesEdge.extraction.edge_extractor:extract_edge_from_frame",
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
    add_batch_policy_argument(parser)


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
    output_base = map_output_path(path, input_path, output_dir).with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    return output_base


def process_extract_file(path: Path, args: argparse.Namespace) -> bool | None:
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
    except (IndexError, OSError, ValueError) as error:
        print(f"Failed to extract {_display_path(path)}: {error}")
        return False
    return True


def run(args: argparse.Namespace) -> int:
    """Run extraction over the selected ND2 files."""
    paths, input_root = select_input_files(args.input_path, ".nd2", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .nd2 files found for {args.input_path}")
    args.input_path = input_root

    failed = succeeded = skipped = processed = 0
    for path in paths:
        processed += 1
        result = process_extract_file(path, args)
        if result is False:
            failed += 1
            if getattr(args, "error_policy", "keep-going") == "fail-fast":
                break
        elif result is None:
            skipped += 1
        else:
            succeeded += 1
    report_batch_summary(processed, succeeded, skipped, failed)
    return exit_code(failed, succeeded)
