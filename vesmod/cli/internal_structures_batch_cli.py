"""Batch-input wrapper for the internal-structures CLI implementation."""

from __future__ import annotations

from pathlib import Path

from . import internal_structures_cli
from .input_selection import select_input_files


def add_parser(subparsers) -> None:
    """Add internal-structures arguments plus additional input selectors."""
    internal_structures_cli.add_parser(subparsers)
    parser = subparsers.choices["internal-structures"]
    parser.add_argument(
        "additional_input_paths",
        type=Path,
        nargs="*",
        help="Additional .npz files, directories, or glob patterns.",
    )


def run(args) -> None:
    """Resolve batch selectors and run the existing measurement orchestration."""
    selectors = [args.input_path, *args.additional_input_paths]
    paths, input_root = select_input_files(selectors, ".npz", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npz files found for {selectors}")
    args.input_path = input_root

    internal_structures_cli._validate_input_output_paths(
        args.input_path,
        args.output_dir,
    )
    config = internal_structures_cli.config_from_args(args)
    qc_config, qc_provenance_path = internal_structures_cli._load_qc_selection(
        args,
        paths,
    )
    internal_structures_cli._write_provenance(
        args,
        paths,
        config,
        qc_config,
        qc_provenance_path,
    )
    video_index = internal_structures_cli._build_video_filename_index(
        paths,
        args.video_root,
    )
    summary_rows = [
        internal_structures_cli.process_checkpoint(
            path,
            args,
            config,
            qc_config,
            video_index,
        )
        for path in paths
    ]
    internal_structures_cli._write_csv(
        args.output_dir / "internal_structure_summary.csv",
        summary_rows,
        internal_structures_cli._SUMMARY_FIELDS,
    )
