"""Command line tool for fitting bending moduli from vesicle edge arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

from vesmod.EdgeMod import (
    FixedFitRangeSelector,
    QMinusThreeFitRangeSelector,
    Spectrum,
    SpectrumFitConfig,
)


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
        help=(
            "Lowest Fourier mode included in a fixed fit or eligible for "
            "dynamic selection. Default: 3."
        ),
    )
    parser.add_argument(
        "--upper-fitting-bound",
        type=int,
        default=8,
        help=(
            "First Fourier mode excluded from a fixed fit or dynamic-search "
            "interval. Default: 8."
        ),
    )
    parser.add_argument(
        "--dynamic-range",
        action="store_true",
        help="Select the fit range from q^-3 scaling instead of using fixed bounds.",
    )
    parser.add_argument(
        "--min-modes",
        type=int,
        help="Minimum consecutive q modes required for dynamic selection.",
    )
    parser.add_argument(
        "--slope-tolerance",
        type=float,
        help="Maximum allowed absolute deviation of the fitted slope from -3.",
    )
    parser.add_argument(
        "--max-log-rmse",
        type=float,
        help="Maximum allowed log-space RMSE to a fixed q^-3 model.",
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


def build_fit_config(args: argparse.Namespace) -> SpectrumFitConfig:
    """Build the scientific fit configuration requested on the command line."""
    if args.dynamic_range:
        missing = [
            option
            for option, value in (
                ("--min-modes", args.min_modes),
                ("--slope-tolerance", args.slope_tolerance),
                ("--max-log-rmse", args.max_log_rmse),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Dynamic range selection requires " + ", ".join(missing) + "."
            )
        range_selector = QMinusThreeFitRangeSelector(
            lower_bound=args.lower_fitting_bound,
            upper_bound=args.upper_fitting_bound,
            min_modes=args.min_modes,
            slope_tolerance=args.slope_tolerance,
            max_log_rmse=args.max_log_rmse,
        )
    else:
        range_selector = FixedFitRangeSelector(
            lower_bound=args.lower_fitting_bound,
            upper_bound=args.upper_fitting_bound,
        )

    return SpectrumFitConfig(
        lmax=args.lmax,
        free_sigma=not args.fixed_sigma,
        temperature=args.temperature,
        range_selector=range_selector,
    )


def output_path_for(path: Path, dynamic_range: bool) -> Path:
    """Return a non-colliding JSON path for the requested fit strategy."""
    if dynamic_range:
        return path.with_name(f"{path.stem}.dynamic.json")
    return path.with_suffix(".json")


def process_file(path: Path, args: argparse.Namespace) -> None:
    """Fit one edge file and save its spectrum metadata to JSON."""
    config = build_fit_config(args)
    output_path = output_path_for(path, args.dynamic_range)

    print(f"Working on file {path.stem}")
    spectrum = Spectrum(path)
    fit = spectrum.extract_kc_from_fit(config)
    print(f"kc={fit.kC}, sigma={fit.surface_tension}")
    spectrum.to_json(output_path)


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
