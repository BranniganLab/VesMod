"""Command line tool for fitting bending moduli from vesicle edge arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

from vesmod.EdgeMod import Spectrum


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit kc/sigma from one or more .npy edge files and write one JSON "
            "output per input file."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A .npy file or a directory containing .npy files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when input_path is a directory.",
    )
    parser.add_argument(
        "--lower-fitting-bound",
        type=int,
        default=3,
        help="Lowest Fourier mode included in the fit. Default: 3.",
    )
    parser.add_argument(
        "--upper-fitting-bound",
        type=int,
        default=8,
        help="First Fourier mode excluded from the fit. Default: 8.",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=500,
        help="Maximum summation index used by the fit. Default: 500.",
    )
    parser.add_argument(
        "--fixed-sigma",
        action="store_true",
        help="Do not fit surface tension sigma as a free parameter.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=295,
        help="Temperature in Kelvin when experiment performed. Default: 295.",
    )
    return parser.parse_args()


def iter_npy_files(input_path: Path, recursive: bool) -> list[Path]:
    """Return the .npy files selected by the user."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix != ".npy":
            raise ValueError(f"Expected a .npy file, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = "**/*.npy" if recursive else "*.npy"
    return sorted(input_path.glob(pattern))


def process_file(path: Path, args: argparse.Namespace) -> None:
    """Fit one edge file and save its spectrum metadata to JSON."""
    output_path = path.with_suffix(".json")

    print(f"Working on file {path.stem}")
    spectrum = Spectrum(path)
    kc, sigma = spectrum.extract_kc_from_fit(
        lower_bound=args.lower_fitting_bound,
        upper_bound=args.upper_fitting_bound,
        lmax=args.lmax,
        free_sigma=not args.fixed_sigma,
        temperature=args.temperature
    )
    print(f"kc={kc}, sigma={sigma}")
    spectrum.to_json(path)


def main() -> None:
    """Run the command line interface."""
    args = parse_args()
    paths = iter_npy_files(args.input_path, args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {args.input_path}")

    for path in paths:
        process_file(path, args)


if __name__ == "__main__":
    main()
