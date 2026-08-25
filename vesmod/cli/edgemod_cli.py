"""CLI for EdgeMod fitting and separate experimental screening stages.

Core EdgeMod always performs a physical fit over fixed q bounds stored in a
``SpectrumFitConfig``. When ``--dynamic-range`` is requested, the CLI first
runs the experimental q^-3 selector, then passes the selected bounds into an
ordinary core fit configuration.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from vesmod.EdgeMod import Spectrum, SpectrumFitConfig
from vesmod.EdgeMod.experimental import (
    DynamicRangeSelection,
    QMinusThreeRangeSelector,
    TemporalRMSConfig,
    TemporalRMSResult,
    calculate_temporal_rms,
)
from vesmod.EdgeMod.experimental.temporal_rms_plotting import (
    save_temporal_rms_histogram,
)


def parse_args() -> argparse.Namespace:
    """Parse the core CLI or an explicitly namespaced experimental command."""
    if len(sys.argv) > 1 and sys.argv[1] == "experimental":
        parser = _experimental_parser()
        return parser.parse_args(sys.argv[2:])

    parser = argparse.ArgumentParser(
        description=(
            "Fit membrane mechanical parameters from vesicle contour "
            "trajectories."
        ),
        epilog=(
            "Experimental analyses are available through "
            "'edgemod experimental --help'."
        ),
    )
    _add_input_options(parser)
    _add_fit_options(parser)
    args = parser.parse_args()
    args.command = "fit"
    return args


def _experimental_parser() -> argparse.ArgumentParser:
    """Return the parser for explicitly experimental EdgeMod analyses."""
    parser = argparse.ArgumentParser(
        prog="edgemod experimental",
        description="Run optional experimental EdgeMod analyses.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_temporal_rms_parser(subparsers)
    return parser


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    """Add input-path and recursion options shared by EdgeMod stages."""
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


def _add_fit_options(parser: argparse.ArgumentParser) -> None:
    """Add options for core physical fitting."""
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


def _add_temporal_rms_parser(subparsers) -> None:
    """Add options for experimental temporal-RMS screening."""
    parser = subparsers.add_parser(
        "temporal-rms",
        help="EXPERIMENTAL: measure temporal contour motion and screen inputs.",
    )
    _add_input_options(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for accepted .npy files, temporal_rms_qc.json, "
            "temporal_rms_summary.csv, and the population histogram."
        ),
    )
    parser.add_argument(
        "--lower-bound",
        type=int,
        default=3,
        help="Lowest Fourier mode included in temporal RMS. Default: 3.",
    )
    parser.add_argument(
        "--upper-bound",
        type=int,
        default=8,
        help="First Fourier mode excluded from temporal RMS. Default: 8.",
    )
    parser.add_argument(
        "--cutoff-nm",
        type=float,
        help=(
            "Minimum included temporal RMS amplitude in nm. If omitted, all "
            "successfully measured inputs are retained."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs from a different temporal-RMS configuration.",
    )


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


def _relative_input_path(path: Path, input_path: Path) -> Path:
    """Return one selected file relative to the user-selected input root."""
    resolved_path = path.expanduser().resolve()
    resolved_input = input_path.expanduser().resolve()
    if resolved_path == resolved_input:
        return Path(resolved_path.name)
    return resolved_path.relative_to(resolved_input)


def build_temporal_rms_config(args: argparse.Namespace) -> TemporalRMSConfig:
    """Translate CLI options into an experimental temporal-RMS config."""
    return TemporalRMSConfig(
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        cutoff_nm=args.cutoff_nm,
    )


def _temporal_rms_provenance(
    config: TemporalRMSConfig,
    input_path: Path,
    recursive: bool,
    paths: list[Path],
) -> dict:
    """Return serializable provenance for one temporal-RMS batch."""
    return {
        "experimental_method": "temporal_rms",
        "input_path": str(input_path.expanduser().resolve()),
        "recursive": recursive,
        "input_manifest": [str(path.resolve()) for path in paths],
        "config": config.to_dict(),
        "exported_files": [],
    }


def _reject_overlapping_temporal_rms_paths(
    input_path: Path,
    output_dir: Path,
) -> None:
    """Reject input/output layouts that could mix sources with exports."""
    resolved_input = input_path.expanduser().resolve()
    resolved_output = output_dir.expanduser().resolve()
    if (
        resolved_input == resolved_output
        or resolved_input in resolved_output.parents
        or resolved_output in resolved_input.parents
    ):
        raise ValueError(
            "Temporal-RMS input and output paths must not overlap. "
            "Choose an --output-dir outside the input path."
        )


def _remove_temporal_rms_artifacts(
    output_dir: Path,
    provenance: dict,
) -> None:
    """Remove only files recorded by a validated prior temporal-RMS batch."""
    exported_files = provenance.get("exported_files")
    if (
        provenance.get("experimental_method") != "temporal_rms"
        or not isinstance(exported_files, list)
        or any(not isinstance(path, str) for path in exported_files)
    ):
        raise ValueError(
            "Existing temporal-RMS metadata does not contain a valid export "
            "manifest; refusing to remove files."
        )

    resolved_output = output_dir.expanduser().resolve()
    export_paths = []
    for exported_file in exported_files:
        relative_path = Path(exported_file)
        export_path = (resolved_output / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or relative_path.suffix != ".npy"
            or resolved_output not in export_path.parents
        ):
            raise ValueError(
                "Existing temporal-RMS metadata contains an unsafe export path; "
                "refusing to remove files."
            )
        export_paths.append(export_path)

    for export_path in export_paths:
        if export_path.is_file():
            export_path.unlink()

    for filename in (
        "temporal_rms_qc.json",
        "temporal_rms_summary.csv",
        "temporal_rms_histogram.png",
    ):
        output_path = output_dir / filename
        if output_path.exists():
            output_path.unlink()


def _write_temporal_rms_provenance(
    args: argparse.Namespace,
    config: TemporalRMSConfig,
    paths: list[Path],
) -> None:
    """Write provenance and reject incompatible existing output."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output_dir / "temporal_rms_qc.json"
    provenance = _temporal_rms_provenance(
        config,
        args.input_path,
        args.recursive,
        paths,
    )
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        comparable_existing = (
            dict(existing) if isinstance(existing, dict) else {}
        )
        comparable_existing["exported_files"] = []
        if comparable_existing == provenance:
            return
        if not args.overwrite:
            raise ValueError(
                "Temporal-RMS output directory already contains results from "
                "a different input selection or configuration. Choose another "
                "--output-dir or use --overwrite."
            )
        _remove_temporal_rms_artifacts(args.output_dir, existing)

    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def process_temporal_rms_file(
    path: Path,
    args: argparse.Namespace,
    config: TemporalRMSConfig,
) -> tuple[dict, TemporalRMSResult | None]:
    """Measure one trajectory, export it when included, and summarize it."""
    relative_path = _relative_input_path(path, args.input_path)
    output_path = args.output_dir / relative_path
    try:
        radii_microns = np.load(path)
        result = calculate_temporal_rms(radii_microns, config)
    except (OSError, TypeError, ValueError) as error:
        return (
            {
                "file": str(relative_path),
                "temporal_rms_nm": "",
                "included": False,
                "status": "analysis_error",
                "error": str(error),
            },
            None,
        )

    status = "included" if result.included else "below_cutoff"
    if result.included:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not output_path.exists():
            np.save(output_path, radii_microns)
        else:
            print(f"Keeping existing output for {path.name}: {output_path.name}")

    return (
        {
            "file": str(relative_path),
            "temporal_rms_nm": result.amplitude_nm,
            "included": result.included,
            "status": status,
            "error": "",
        },
        result,
    )


def _record_temporal_rms_exports(
    output_dir: Path,
    rows: list[dict],
) -> None:
    """Record the accepted arrays managed by this temporal-RMS batch."""
    provenance_path = output_dir / "temporal_rms_qc.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["exported_files"] = [
        row["file"] for row in rows if row["included"]
    ]
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_temporal_rms_summary(output_dir: Path, rows: list[dict]) -> None:
    """Write one experimental temporal-RMS summary row per input."""
    summary_path = output_dir / "temporal_rms_summary.csv"
    fieldnames = [
        "file",
        "temporal_rms_nm",
        "included",
        "status",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_fit(args: argparse.Namespace) -> None:
    """Run core fitting over selected contour trajectories."""
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


def _run_temporal_rms(args: argparse.Namespace) -> None:
    """Run one experimental temporal-RMS configuration over selected inputs."""
    _reject_overlapping_temporal_rms_paths(args.input_path, args.output_dir)
    paths = iter_npy_files(args.input_path, args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {args.input_path}")

    config = build_temporal_rms_config(args)
    _write_temporal_rms_provenance(
        args,
        config,
        paths,
    )
    processed = [
        process_temporal_rms_file(path, args, config)
        for path in paths
    ]
    rows = [row for row, _ in processed]
    results = [result for _, result in processed if result is not None]
    _record_temporal_rms_exports(args.output_dir, rows)
    _write_temporal_rms_summary(args.output_dir, rows)
    save_temporal_rms_histogram(
        results,
        args.output_dir / "temporal_rms_histogram.png",
    )


def main() -> None:
    """Run the selected EdgeMod analysis stage."""
    args = parse_args()
    if args.command == "fit":
        _run_fit(args)
    else:
        _run_temporal_rms(args)


if __name__ == "__main__":
    main()
