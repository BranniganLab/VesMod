"""Register and dispatch EdgeMod command-line workflows."""

from __future__ import annotations

import argparse
import sys

from vesmod.cli.edgemod import ensemble, fitting
from vesmod.cli.edgemod.options import (
    add_input_options,
    add_fit_options,
    build_fit_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for core physical fitting.

    Returns
    -------
    argparse.Namespace
        Parsed options for the selected EdgeMod command.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fit membrane mechanical parameters from vesicle contour "
            "trajectories. Use edgemod ensemble --help for ensemble fitting."
        ),
    )
    is_ensemble = sys.argv[1:2] == ["ensemble"]
    if is_ensemble:
        parser.prog += " ensemble"
    add_input_options(parser, ensemble=is_ensemble)
    add_fit_options(parser, ensemble=is_ensemble)
    args = parser.parse_args(sys.argv[2:] if is_ensemble else None)
    args.command = "ensemble" if is_ensemble else "fit"
    return args


def main() -> int:
    """Run the selected EdgeMod analysis stage.

    Returns
    -------
    int
        Exit status of the selected analysis command.
    """
    args = parse_args()
    if getattr(args, "command", "fit") == "ensemble":
        return ensemble.run_ensemble(args, build_fit_config(args))
    return fitting.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
