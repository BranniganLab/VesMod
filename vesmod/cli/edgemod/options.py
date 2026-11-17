"""Shared arguments and fit configuration for EdgeMod commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from vesmod.edgemod import SpectrumFitConfig
from vesmod.edgemod.fitting.mode_relaxation.relaxation import RELAXATION_MODELS
from vesmod.cli.input_selection import InputPathsAction
from vesmod.cli.batch_policy import add_batch_policy_argument

def add_input_options(parser: argparse.ArgumentParser, *, ensemble=False) -> None:
    """Add input-path and recursion options shared by EdgeMod stages.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to configure.
    ensemble : bool
        Whether the command consumes spectrum JSON.

    Returns
    -------
    None
        Registers input selection options.
    """
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="+",
        action=InputPathsAction,
        help=("Saved spectrum JSON files, directories, or glob patterns." if ensemble
              else "One or more .npy files, directories, or glob patterns."),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for directory and glob inputs.",
    )

def add_fit_options(parser: argparse.ArgumentParser, *, ensemble=False) -> None:
    """Add options for core physical fitting.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to configure.
    ensemble : bool
        Whether to register ensemble options.

    Returns
    -------
    None
        Registers fitting options.
    """
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
    if not ensemble:
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
    if ensemble:
        parser.add_argument("--output", type=Path, required=True, help="Ensemble JSON report.")
        parser.add_argument("--exclusion-policy", metavar="MODULE:FUNCTION",
                            help="Importable Python function returning an exclusion reason or None.")
        parser.add_argument("--weight-by-replica-sem", action="store_true")
    else:
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
    parser.add_argument("--model", choices=["spherical_contour_spectrum"],
                        default="spherical_contour_spectrum", help="Equilibrium spectrum model.")
    parser.add_argument("--camera-integration", action="store_true",
                        help="Apply finite camera-exposure averaging to the prediction.")
    parser.add_argument("--relaxation-model", choices=tuple(RELAXATION_MODELS),
                        default="solvent_only", help="Method for spherical-mode relaxation times.")
    parser.add_argument("--exposure-time", type=float, default=0.0,
                        help="Exposure duration in seconds; positive value required with --camera-integration.")
    parser.add_argument("--viscosity-in", type=float, default=None,
                        help="Interior viscosity in Pa s; required for solvent_only camera integration.")
    parser.add_argument("--viscosity-out", type=float, default=None,
                        help="Exterior viscosity in Pa s; common solvent viscosity for solvent_and_membrane.")
    parser.add_argument("--eta-m", type=float, default=None,
                        help="Fixed membrane surface viscosity in Pa s m; required for solvent_and_membrane. "
                             "Does not select the relaxation model.")
    add_batch_policy_argument(parser)

def build_fit_config(args: argparse.Namespace) -> SpectrumFitConfig:
    """Translate CLI options into the core physical-fit configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed physical fit options.

    Returns
    -------
    SpectrumFitConfig
        Validated fitting configuration.
    """
    return SpectrumFitConfig(
        lower_bound=args.lower_fitting_bound,
        upper_bound=args.upper_fitting_bound,
        lmax=args.lmax,
        free_sigma=not args.fixed_sigma,
        temperature=args.temperature,
        exposure_time=getattr(args, "exposure_time", 0.0),
        viscosity_in=getattr(args, "viscosity_in", None),
        viscosity_out=getattr(args, "viscosity_out", None),
        eta_m=getattr(args, "eta_m", None),
        model=getattr(args, "model", "spherical_contour_spectrum"),
        camera_integration=getattr(args, "camera_integration", False),
        relaxation_model=getattr(args, "relaxation_model", "solvent_only"),
    )
