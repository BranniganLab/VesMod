"""Register and dispatch VesEdge command-line workflows."""

from __future__ import annotations

import argparse

from vesmod.VesEdge.cli import extraction_cli, qc_cli
from vesmod.VesEdge.experimental.cli import internal_structures_cli
from vesmod.VesEdge.cli.gif_cli import add_gif_parser, run_gif
from vesmod.VesEdge.cli.trace_plot_cli import add_trace_plot_parser, run_trace_plot


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract vesicle edges to reusable checkpoints, then apply quality "
            "control in a separate stage."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extraction_cli.add_parser(subparsers)
    qc_cli.add_parser(subparsers)
    internal_structures_cli.add_parser(subparsers)
    add_gif_parser(subparsers)
    add_trace_plot_parser(subparsers)
    return parser.parse_args()


def main() -> int:
    """Run the selected VesEdge subcommand."""
    args = parse_args()
    if args.command == "extract":
        return extraction_cli.run(args)
    elif args.command == "qc":
        return qc_cli.run(args)
    elif args.command == "gif":
        return run_gif(args)
    elif args.command == "internal-structures":
        return internal_structures_cli.run(args)
    elif args.command == "plot-traces":
        return run_trace_plot(args)
    else:
        raise ValueError(f"Unknown VesEdge command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
