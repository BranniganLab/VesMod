"""Command line tool for extracting vesicle edges from ND2 videos."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path

import nd2

from vesmod.VesEdge import VesicleVideo


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract vesicle edges from one or more .nd2 files, then save a GIF "
            "and .npy edge file for each video."
        )
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
        "--micron-to-pixel-ratio",
        type=float,
        default=1.0,
        help="Microns per pixel in the microscope image.",
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=5,
        help="Curvature threshold passed to extract_edges. Default: 5.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Do not output a GIF.",
    )
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="If used, downsamples edge extraction outputs to --n-samples evenly-spaced values.",
    )
    parser.add_argument(
        "--n-samples",
        default=120,
        type=int,
        help="If --downsample is used, downsamples edge extraction outputs to --n-samples evenly-spaced values. Default is 120.",
    )
    parser.add_argument(
        "--extractor",
        default="vesmod.VesEdge:extract_edge_from_frame",
        help=(
            "Edge extractor function as 'module:function'. "
            "The function must accept one 2D frame and return (r_vals, vesicle_center). "
            "Default: vesmod.VesEdge:extract_edge_from_frame."
        ),
    )
    parser.add_argument(
        "--extractor-file",
        default=None,
        type=Path,
        help="Path to a Python file containing a custom edge extractor function.",
    )
    parser.add_argument(
        "--extractor-name",
        default="extract_edge_from_frame",
        help="Name of the extractor function in --extractor-file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Process files even when the corresponding .gif already exists.",
    )
    return parser.parse_args()


def iter_nd2_files(input_path: Path, recursive: bool) -> list[Path]:
    """Return the .nd2 files selected by the user."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() != ".nd2":
            raise ValueError(f"Expected an .nd2 file, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*.nd2" if recursive else "*.nd2"
    return sorted(input_path.glob(pattern))


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

    spec = importlib.util.spec_from_file_location("custom_vesedge_extractor", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import extractor file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    extractor = getattr(module, function_name)
    if not callable(extractor):
        raise TypeError(f"{function_name} in {file_path} is not callable.")

    return extractor


def process_file(path: Path, args: argparse.Namespace) -> None:
    """Extract edges from one ND2 video and save standard outputs."""
    output_paths = [path.with_suffix(".npy")]

    if not args.no_gif:
        output_paths.append(path.with_suffix(".gif"))

    existing_paths = [output_path for output_path in output_paths if output_path.exists()]

    if existing_paths and not args.overwrite:
        existing_names = ", ".join(output_path.name for output_path in existing_paths)
        print(f"Skipping {path.name}: output file(s) already exist: {existing_names}")
        return

    if args.extractor_file is not None:
        extractor_func = load_extractor_from_file(
            args.extractor_file,
            args.extractor_name,
        )
    else:
        extractor_func = load_extractor_from_module(args.extractor)

    print(f"Working on file {path.stem}")
    intensities = nd2.imread(path)
    if args.downsample:
        video = VesicleVideo(intensities, args.micron_to_pixel_ratio, args.n_samples)
    else:
        video = VesicleVideo(intensities, args.micron_to_pixel_ratio, None)

    video.extract_edges(
        extractor_func,
        curvature_threshold=args.curvature_threshold,
    )
    if not args.no_gif:
        video.make_vesicle_gif(path)
    video.save_edge_to_npy(path)


def main() -> None:
    """Run the command line interface."""
    args = parse_args()
    paths = iter_nd2_files(args.input_path, args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .nd2 files found in {args.input_path}")

    for path in paths:
        process_file(path, args)


if __name__ == "__main__":
    main()
