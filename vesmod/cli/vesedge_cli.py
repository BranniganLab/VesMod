"""Command line tool for extracting vesicle edges from ND2 videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import nd2

from vesmod.VesEdge import VesicleVideo, extract_edge_from_frame


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
        default=1 / 13.44,
        help="Microns per pixel in the microscope image. Default: 1/13.44.",
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=5,
        help="Curvature threshold passed to extract_edges. Default: 10.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Do not output a GIF.",
    )
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="If used, downsamples edge extraction outputs to --n_samples evenly-spaced values.",
    )
    parser.add_argument(
        "--n_samples",
        default=120,
        help="If --downsample is used, downsamples edge extraction outputs to --n_samples evenly-spaced values. Default is 120.",
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


def process_file(path: Path, args: argparse.Namespace) -> None:
    """Extract edges from one ND2 video and save standard outputs."""
    gif_path = path.with_suffix(".gif")
    if gif_path.exists() and not args.overwrite:
        print(f"Skipping {path.name}: {gif_path.name} already exists")
        return

    print(f"Working on file {path.stem}")
    intensities = nd2.imread(path)
    if args.downsample:
        video = VesicleVideo(intensities, args.micron_to_pixel_ratio, args.n_samples)
    else:
        video = VesicleVideo(intensities, args.micron_to_pixel_ratio, None)

    video.extract_edges(
        extract_edge_from_frame,
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
