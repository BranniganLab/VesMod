"""CLI for EdgeMod fitting and separate experimental screening stages.

Core EdgeMod always performs a physical fit over fixed q bounds stored in a
``SpectrumFitConfig``. When ``--dynamic-range`` is requested, the CLI first
runs the experimental q^-3 selector, then passes the selected bounds into an
ordinary core fit configuration.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
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

from vesmod.cli.input_selection import InputPathsAction, select_input_files
from vesmod.cli.path_utils import remove_manifest_artifacts


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
        nargs="+",
        action=InputPathsAction,
        help="One or more .npy files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for directory and glob inputs.",
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for fit JSON, diagnostics, provenance, and a batch "
            "summary. By default, outputs are written beside each input file."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs managed by a prior compatible EdgeMod batch.",
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


def iter_npy_files(input_path: Path | list[Path], recursive: bool) -> list[Path]:
    """Return the input ``.npy`` files selected by CLI selectors."""
    paths, _ = select_input_files(input_path, ".npy", recursive)
    return paths


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

    data = spectrum.to_dict(include_arrays=True)
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


def _fit_output_path(path: Path, args: argparse.Namespace) -> Path:
    """Return the fit JSON path, optionally beneath an external output root."""
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        return output_path_for(path, args.dynamic_range)
    relative_path = _relative_input_path(path, args.input_path)
    output_input = output_dir / relative_path
    output_input.parent.mkdir(parents=True, exist_ok=True)
    return output_path_for(output_input, args.dynamic_range)


def process_file(path: Path, args: argparse.Namespace):
    """Fit one trajectory and save core results plus optional diagnostics."""
    config = build_fit_config(args)
    output_path = _fit_output_path(path, args)
    diagnostic_path = output_path.with_suffix(".spectrum_diagnostic.png")
    if (
        getattr(args, "output_dir", None) is not None
        and output_path.exists()
        and not getattr(args, "overwrite", False)
    ):
        print(f"Keeping existing output for {path}: {output_path}")
        return None

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
    return fit


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
    if not isinstance(provenance, dict) or provenance.get(
        "experimental_method"
    ) != "temporal_rms":
        raise ValueError(
            "Existing temporal-RMS metadata is invalid; refusing to remove files."
        )
    remove_manifest_artifacts(
        output_dir,
        provenance,
        manifest_key="exported_files",
        manifest_name="temporal-RMS metadata",
        allowed_suffixes={".npy"},
        metadata_files=(
            "temporal_rms_qc.json",
            "temporal_rms_summary.csv",
            "temporal_rms_histogram.png",
        ),
    )


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


def _reject_overlapping_fit_paths(
    input_roots: tuple[Path, ...],
    output_dir: Path,
) -> None:
    """Reject external fit outputs that overlap an actual selected input tree."""
    resolved_output = output_dir.expanduser().resolve()
    for input_root in input_roots:
        resolved_input = input_root.expanduser().resolve()
        if (
            resolved_input == resolved_output
            or resolved_input in resolved_output.parents
            or resolved_output in resolved_input.parents
        ):
            raise ValueError(
                "EdgeMod input and output paths must not overlap. "
                "Choose an --output-dir outside the input path."
            )


def _fit_provenance(args: argparse.Namespace, paths: list[Path]) -> dict:
    """Return reproducible provenance for one external-output fit batch."""
    config = asdict(build_fit_config(args))
    dynamic = None
    if args.dynamic_range:
        dynamic = {
            "min_modes": args.min_modes,
            "slope_tolerance": args.slope_tolerance,
            "max_log_rmse": args.max_log_rmse,
        }
    return {
        "analysis": "edgemod_fit",
        "input_path": str(args.input_path.expanduser().resolve()),
        "recursive": args.recursive,
        "input_manifest": [str(path.resolve()) for path in paths],
        "fit_config": config,
        "dynamic_range": args.dynamic_range,
        "dynamic_range_config": dynamic,
        "managed_artifacts": [],
    }


def _remove_fit_artifacts(output_dir: Path, provenance: dict) -> None:
    """Remove only artifacts recorded by a validated prior fit batch."""
    if not isinstance(provenance, dict) or provenance.get("analysis") != "edgemod_fit":
        raise ValueError(
            "Existing EdgeMod provenance has no valid artifact manifest; "
            "refusing to remove files."
        )
    remove_manifest_artifacts(
        output_dir,
        provenance,
        manifest_key="managed_artifacts",
        manifest_name="EdgeMod provenance",
        allowed_suffixes={".json", ".png"},
        metadata_files=("edgemod_fit.json", "fit_summary.csv"),
    )


def _prepare_fit_output(args: argparse.Namespace, paths: list[Path]) -> None:
    """Validate and write provenance for an external-output fit batch."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output_dir / "edgemod_fit.json"
    provenance = _fit_provenance(args, paths)
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        comparable = dict(existing) if isinstance(existing, dict) else {}
        comparable["managed_artifacts"] = []
        if comparable == provenance:
            if args.overwrite:
                _remove_fit_artifacts(args.output_dir, existing)
            else:
                provenance["managed_artifacts"] = existing.get(
                    "managed_artifacts",
                    [],
                )
        else:
            if not args.overwrite:
                raise ValueError(
                    "EdgeMod output directory already contains results from a "
                    "different input selection or fit configuration. Choose "
                    "another --output-dir or use --overwrite."
                )
            _remove_fit_artifacts(args.output_dir, existing)
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def _fit_summary_row(
    path: Path,
    input_root: Path,
    status: str,
    fit=None,
    error: str = "",
) -> dict:
    """Return one batch-summary row."""
    return {
        "file": str(_relative_input_path(path, input_root)),
        "status": status,
        "kC": "" if fit is None else fit.kC,
        "surface_tension": "" if fit is None else fit.surface_tension,
        "error": error,
    }


def _safe_csv_text(value):
    """Prefix spreadsheet-formula-like text while preserving other values."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_fit_batch_outputs(
    args: argparse.Namespace,
    rows: list[dict],
) -> None:
    """Write the summary and record JSON/PNG files owned by this batch."""
    summary_path = args.output_dir / "fit_summary.csv"
    safe_rows = [
        {
            **row,
            "file": _safe_csv_text(row["file"]),
            "error": _safe_csv_text(row["error"]),
        }
        for row in rows
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "status", "kC", "surface_tension", "error"],
        )
        writer.writeheader()
        writer.writerows(safe_rows)

    provenance_path = args.output_dir / "edgemod_fit.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    artifacts = []
    for row in rows:
        relative_input = Path(row["file"])
        json_path = output_path_for(
            args.output_dir / relative_input,
            args.dynamic_range,
        )
        diagnostic_path = json_path.with_suffix(".spectrum_diagnostic.png")
        for artifact in (json_path, diagnostic_path):
            if artifact.is_file():
                artifacts.append(str(artifact.relative_to(args.output_dir)))
    provenance["managed_artifacts"] = sorted(artifacts)
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_fit(args: argparse.Namespace) -> None:
    """Run core fitting over selected contour trajectories."""
    build_fit_config(args)
    if args.dynamic_range:
        build_dynamic_selector(args)

    paths, input_root, input_roots = select_input_files(
        args.input_path,
        ".npy",
        args.recursive,
        return_roots=True,
    )
    if not paths:
        raise FileNotFoundError(f"No .npy files found for {args.input_path}")
    args.input_path = input_root
    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        _reject_overlapping_fit_paths(input_roots, output_dir)
        _prepare_fit_output(args, paths)

    rows = []
    for path in paths:
        try:
            fit = process_file(path, args)
            if output_dir is not None:
                status = "kept_existing" if fit is None else "ok"
                rows.append(
                    _fit_summary_row(path, args.input_path, status, fit=fit)
                )
        except (OSError, ValueError, FloatingPointError) as exc:
            if output_dir is not None:
                rows.append(
                    _fit_summary_row(
                        path,
                        args.input_path,
                        "fit_error",
                        error=str(exc),
                    )
                )
            if not args.recursive:
                if output_dir is not None:
                    _write_fit_batch_outputs(args, rows)
                raise
            print(f"Skipping {path}: {exc}", file=sys.stderr)

    if output_dir is not None:
        _write_fit_batch_outputs(args, rows)


def _run_temporal_rms(args: argparse.Namespace) -> None:
    """Run one experimental temporal-RMS configuration over selected inputs."""
    paths, input_root = select_input_files(args.input_path, ".npy", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npy files found for {args.input_path}")
    args.input_path = input_root
    _reject_overlapping_temporal_rms_paths(args.input_path, args.output_dir)

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
