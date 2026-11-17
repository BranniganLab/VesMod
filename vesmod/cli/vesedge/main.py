"""Register and dispatch VesEdge command-line workflows."""

from __future__ import annotations

import argparse

from vesmod.cli.vesedge import extraction, gif, qc, trace_plot
from vesmod.cli.vesedge.experimental import internal_structures


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract vesicle edges to reusable checkpoints, then apply quality "
            "control in a separate stage."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extraction.add_parser(subparsers)
    qc.add_parser(subparsers)
    internal_structures.add_parser(subparsers)
    gif.add_parser(subparsers)
    trace_plot.add_parser(subparsers)
    return parser.parse_args()


def main() -> int:
    """Run the selected VesEdge subcommand."""
    args = parse_args()
    if args.command == "extract":
        return extraction.run(args)
    elif args.command == "qc":
        return qc.run(args)
    elif args.command == "gif":
        return gif.run(args)
    elif args.command == "internal-structures":
        return internal_structures.run(args)
    elif args.command == "plot-traces":
        return trace_plot.run(args)
    else:
        raise ValueError(f"Unknown VesEdge command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
