"""CLI for core EdgeMod fitting with optional experimental range selection.

Core EdgeMod always performs a physical fit over fixed q bounds stored in a
``SpectrumFitConfig``. When ``--dynamic-range`` is requested, the CLI first
runs the experimental q^-3 selector, then passes the selected bounds into an
ordinary core fit configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from vesmod.EdgeMod import Spectrum, SpectrumFitConfig
from vesmod.EdgeMod.experimental import (
    DynamicRangeSelection,
    QMinusThreeRangeSelector,
)


def parse_args() -> argparse.Namespace:
    """Parse input-selection, physical-fit, and experimental options."""
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
            "Lowest Fourier mode used by a fixed fit or eligible for the "
            "experimental dynamic search. Default: 3."
        ),
    )
    parser.add_argument(
        "--upper-fitting-bound",
        type=int,
        default=8,
        help=(
            "First Fourier mode excluded from a fixed fit or experimental "
            "dynamic search. Default: 8."
        ),
    )
    parser.add_argument(
        "--dynamic-range",
        action="store_true",
        help="EXPERIMENTAL: choose fit bounds from q^-3 scaling.",
    )
    parser.add_argument(
        "--min-modes",
        type=int,
        help="EXPERIMENTAL: minimum consecutive q modes for dynamic selection.",
    )
    parser.add_argument(
        "--slope-tolerance",
        type=float,
        help="EXPERIMENTAL: maximum deviation of fitted slope from -3.",
    )
    parser.add_argument(
        "--max-log-rmse",
        type=float,
        help="EXPERIMENTAL: maximum log-space RMSE to a fixed q^-3 model.",
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
    """Return the input ``.npy`` files selected by a CLI path and recursion flag."""
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
    """Translate CLI options into the core physical-fit configuration."""
    return SpectrumFitConfig(
        lower_bound=args.lower_fitting_bound,
        upper_bound=args.upper_fitting_bound,
        lmax=args.lmax,
        free_sigma=not args.fixed_sigma,
        temperature=args.temperature,
    )


def build_dynamic_selector(args: argparse.Namespace) -> QMinusThreeRangeSelector:
    """Build the explicitly requested experimental q^-3 selector."""
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
    return QMinusThreeRangeSelector(
        lower_bound=args.lower_fitting_bound,
        upper_bound=args.upper_fitting_bound,
        min_modes=args.min_modes,
        slope_tolerance=args.slope_tolerance,
        max_log_rmse=args.max_log_rmse,
    )


def output_path_for(path: Path, dynamic_range: bool) -> Path:
    """Return the JSON output path, separating experimental from core results."""
    if dynamic_range:
        return path.with_name(f"{path.stem}.dynamic.json")
    return path.with_suffix(".json")


def _write_output(
    spectrum: Spectrum,
    output_path: Path,
    selection: DynamicRangeSelection | None = None,
) -> None:
    """Serialize core spectrum state plus optional experimental diagnostics."""
    if selection is None:
        spectrum.to_json(output_path)
        return

    data = spectrum._to_dict(include_arrays=True)
    data["experimental"] = {
        "dynamic_range_selection": selection.to_dict(),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _select_dynamic_config(
    spectrum: Spectrum,
    config: SpectrumFitConfig,
    selector: QMinusThreeRangeSelector,
) -> tuple[SpectrumFitConfig, DynamicRangeSelection]:
    """Run experimental selection and return a core config with chosen bounds."""
    selection = selector.select(spectrum.modes, spectrum.avg_amps2)
    if not selection.accepted:
        return config, selection
    if selection.lower_bound is None or selection.upper_bound is None:
        raise ValueError("Accepted dynamic selection is missing q bounds.")
    return (
        replace(
            config,
            lower_bound=selection.lower_bound,
            upper_bound=selection.upper_bound,
        ),
        selection,
    )


def process_file(path: Path, args: argparse.Namespace) -> None:
    """Fit one trajectory and save core results plus optional diagnostics."""
    config = build_fit_config(args)
    output_path = output_path_for(path, args.dynamic_range)
    diagnostic_path = output_path.with_suffix(".spectrum_diagnostic.png")

    print(f"Working on file {path.stem}")
    spectrum = Spectrum(path)
    selection = None

    if args.dynamic_range:
        selector = build_dynamic_selector(args)
        config, selection = _select_dynamic_config(spectrum, config, selector)
        if not selection.accepted:
            _write_output(spectrum, output_path, selection)
            raise ValueError(selection.reason or "Dynamic q-range selection failed.")

    try:
        fit = spectrum.extract_kc_from_fit(config)
    except ValueError as error:
        _write_output(spectrum, output_path, selection)
        if spectrum.fit_result is not None:
            spectrum.save_fit_diagnostic(
                diagnostic_path,
                config.lower_bound,
                config.upper_bound,
                config.lmax,
                validation_error=str(error),
            )
        raise

    spectrum.save_fit_diagnostic(
        diagnostic_path,
        fit.lower_bound,
        fit.upper_bound,
        config.lmax,
    )

    print(f"kc={fit.kC}, sigma={fit.surface_tension}")
    _write_output(spectrum, output_path, selection)


def main() -> None:
    """Run EdgeMod over the selected file or batch of files."""
    args = parse_args()
    build_fit_config(args)
    if args.dynamic_range:
        build_dynamic_selector(args)

    paths = iter_npy_files(args.input_path, args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {args.input_path}")

    for path in paths:
        try:
            process_file(path, args)
        except (ValueError, FloatingPointError) as exc:
            if not args.recursive:
                raise
            print(f"Skipping {path}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
