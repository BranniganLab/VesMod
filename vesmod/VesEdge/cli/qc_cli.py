"""Command-line quality control of VesEdge extraction checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vesmod.VesEdge.io.checkpoint_sources import open_checkpoint_frames, resolve_source_path
from vesmod.VesEdge.qc import qc_runner, qc_registry
from vesmod.VesEdge.qc.qc_config import VesicleQCConfig
from vesmod.VesEdge.qc.acceptance_rules import AnyFailed, AllFailed
from vesmod.VesEdge.vesicle_edges import VesicleEdges
from vesmod.cli.batch_policy import (
    add_batch_policy_argument,
    exit_code,
    report_batch_summary,
)
from vesmod.cli.input_selection import InputPathsAction, select_input_files
from vesmod.cli.path_utils import (
    _display_path,
    _relative_input_path,
    remove_manifest_artifacts,
)
from vesmod.io.path_mapping import map_output_path


def add_parser(subparsers) -> None:
    """Add arguments for the QC stage."""
    parser = subparsers.add_parser(
        "qc",
        help="Apply QC to .npz checkpoints and save filtered .npy files.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="+",
        action=InputPathsAction,
        help="One or more VesEdge .npz files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for directory and glob inputs.",
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
    rules = parser.add_mutually_exclusive_group()
    names = tuple(check.name for check in qc_registry.QC_CHECKS)
    rules.add_argument(
        "--reject-if-any-fail", nargs="+", choices=names, metavar="CHECK",
        help="Reject a frame if any named check fails; all named checks must be enabled.",
    )
    rules.add_argument(
        "--reject-if-all-fail", nargs="+", choices=names, metavar="CHECK",
        help="Reject a frame only if all named checks fail; all named checks must be enabled.",
    )
    add_batch_policy_argument(parser)
    for check in qc_registry.QC_CHECKS:
        check.add_cli_arguments(parser)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing filtered .npy outputs and QC provenance.",
    )


def _qc_config_from_args(args: argparse.Namespace) -> VesicleQCConfig:
    """Build the QC configuration requested on the command line."""
    any_checks = getattr(args, "reject_if_any_fail", None)
    all_checks = getattr(args, "reject_if_all_fail", None)
    rule = None
    if any_checks:
        rule = AnyFailed(*any_checks)
    elif all_checks:
        rule = AllFailed(*all_checks)
    return VesicleQCConfig(
        checks={check.name: check.config_from_args(args) for check in qc_registry.QC_CHECKS},
        reject_when=rule,
    )


def _qc_provenance(
    qc_config: VesicleQCConfig,
    input_path: Path,
    recursive: bool,
    paths: list[Path],
) -> dict:
    """Return serializable provenance for one resolved QC batch."""
    provenance = {
        "input_path": str(input_path.expanduser().resolve()),
        "recursive": recursive,
        "checkpoint_manifest": [str(path.resolve()) for path in paths],
        "qc_config": qc_config.to_dict(),
        "managed_artifacts": [],
    }
    return provenance


def _remove_managed_qc_artifacts(output_dir: Path) -> None:
    """Remove filtered arrays and metadata managed by a previous QC batch."""
    provenance_path = output_dir / "vesedge_qc.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            "Existing VesEdge QC provenance is malformed; refusing to remove "
            "files."
        ) from error
    if not isinstance(provenance, dict) or not {
        "checkpoint_manifest",
        "qc_config",
    }.issubset(provenance):
        raise ValueError(
            "Existing VesEdge QC provenance is incomplete; refusing to remove "
            "files."
        )
    remove_manifest_artifacts(
        output_dir,
        provenance,
        manifest_key="managed_artifacts",
        manifest_name="VesEdge QC provenance",
        allowed_suffixes={".npy", ".png", ".csv"},
        metadata_files=("qc_summary.csv", "vesedge_qc.json"),
    )


def _write_qc_provenance(
    output_dir: Path,
    qc_config: VesicleQCConfig,
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
        comparable = dict(existing) if isinstance(existing, dict) else {}
        comparable["managed_artifacts"] = []
        if comparable == provenance:
            if overwrite:
                _remove_managed_qc_artifacts(output_dir)
                # The matching-provenance cleanup already removed the old
                # manifest; continue by writing the replacement below.
            else:
                provenance["managed_artifacts"] = existing.get(
                    "managed_artifacts", []
                )
                provenance_path.write_text(
                    json.dumps(provenance, indent=2) + "\n",
                    encoding="utf-8",
                )
                return
        else:
            if not overwrite:
                raise ValueError(
                    "QC output directory already contains results from a different "
                    "input selection or QC configuration. Choose another "
                    "--output-dir or use --overwrite."
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
    accepted = (
        0 if edges.qc_result is None else len(edges.qc_result.accepted_frame_indices)
    )
    return {
        "file": str(_relative_input_path(path, input_path)),
        "frames": len(edges.detections),
        "successful_detections": len(successful),
        "extraction_failures": len(edges.detections) - len(successful),
        **_check_summary(edges),
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
        **_check_summary(),
        "accepted": 0,
        "accepted_fraction": 0.0,
        "status": "load_error",
        "error": error,
    }


def process_qc_file(
    path: Path,
    args: argparse.Namespace,
    qc_config: VesicleQCConfig,
    managed_artifacts: set[Path] | None = None,
) -> dict:
    """Apply QC to one checkpoint and return its batch summary row."""
    output_path = map_output_path(path, args.input_path, args.output_dir, suffix=".npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_exists = output_path.exists()
    if output_exists and not args.overwrite:
        print(
            f"Keeping existing output for {_display_path(path)}: "
            f"{_display_path(output_path)}"
        )

    try:
        edges = VesicleEdges.from_checkpoint(path)
    except (FileNotFoundError, ValueError) as error:
        message = str(error)
        print(f"Failed to load {_display_path(path)}: {message}")
        return _load_error_summary(path, args.input_path, message)

    status = "ok"
    qc_error = ""
    try:
        frames = None
        frame_checks = qc_runner.checks_requiring_source_video_frames(qc_config)
        if frame_checks:
            if edges.source_path is None:
                names = ", ".join(check.name.replace("_", "-") for check in frame_checks)
                raise ValueError(
                    f"{names.capitalize()} QC requires a checkpoint with a source video path."
                )
            source_path = resolve_source_path(edges.source_path, path)
            frames = open_checkpoint_frames(source_path)
        if frames is None:
            edges.run_qc(qc_config)
        else:
            with frames:
                edges.run_qc(qc_config, frames=frames)
    except (OSError, ValueError) as error:
        qc_error = str(error)
        if edges.qc_result is None:
            status = "qc_error"
            print(f"QC failed for {_display_path(path)}: {error}")
        else:
            status = "no_accepted_frames"
            print(
                "QC produced no accepted frames for "
                f"{_display_path(path)}: {error}"
            )

    if status == "no_accepted_frames" and args.overwrite and output_exists:
        output_path.unlink()

    row = _qc_summary(
        path,
        args.input_path,
        edges,
        status,
        qc_error,
    )
    if edges.qc_result is not None:
        for check in qc_registry.QC_CHECKS:
            if edges.qc_result.get_result(check.name) is None:
                continue
            for suffix, write in check.diagnostic_writers:
                diagnostic_path = output_path.with_suffix(suffix)
                if args.overwrite or not diagnostic_path.exists():
                    write(diagnostic_path, edges)
                    if managed_artifacts is not None:
                        managed_artifacts.add(diagnostic_path)
    if (
        status == "ok"
        and row["accepted"] > 0
        and (args.overwrite or not output_exists)
    ):
        edges.export_accepted_radii(output_path)
        if managed_artifacts is not None:
            managed_artifacts.add(output_path)
    return row


def _record_qc_artifacts(
    output_dir: Path,
    managed_artifacts: set[Path],
) -> None:
    """Record the filtered arrays and diagnostics created by this batch."""
    provenance_path = output_dir / "vesedge_qc.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["managed_artifacts"] = sorted(
        str(path.resolve().relative_to(output_dir.resolve()))
        for path in managed_artifacts
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def _check_summary(edges: VesicleEdges | None = None) -> dict:
    """Collect registered check columns and, when available, their values."""
    values = {}
    for check in sorted(qc_registry.QC_CHECKS, key=lambda check: check.summary_order):
        values.update(check.summary_defaults)
        if edges is not None:
            values.update(check.summary_values(edges))
    return values


def _write_qc_summary(output_dir: Path, rows: list[dict]) -> None:
    """Write one CSV row per processed checkpoint."""
    summary_path = output_dir / "qc_summary.csv"
    fieldnames = [
        "file",
        "frames",
        "successful_detections",
        "extraction_failures",
        *_check_summary(),
        "accepted",
        "accepted_fraction",
        "status",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    """Run one QC configuration over the selected checkpoints."""
    paths, input_root = select_input_files(args.input_path, ".npz", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npz files found for {args.input_path}")
    args.input_path = input_root

    qc_config = _qc_config_from_args(args)
    _write_qc_provenance(
        args.output_dir,
        qc_config,
        args.input_path,
        args.recursive,
        paths,
        args.overwrite,
    )
    managed_artifacts: set[Path] = set()
    rows = []
    for path in paths:
        row = process_qc_file(path, args, qc_config, managed_artifacts)
        rows.append(row)
        if row["status"] in {"load_error", "qc_error", "no_accepted_frames"} and getattr(args, "error_policy", "keep-going") == "fail-fast":
            break
    _write_qc_summary(args.output_dir, rows)
    _record_qc_artifacts(args.output_dir, managed_artifacts)
    failed = sum(
        row["status"] in {"load_error", "qc_error", "no_accepted_frames"}
        for row in rows
    )
    succeeded = len(rows) - failed
    report_batch_summary(len(rows), succeeded, 0, failed)
    return exit_code(failed, succeeded)
