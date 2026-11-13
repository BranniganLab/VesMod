"""Fit an ensemble and record the selection of replica spectra."""

import json
from importlib import import_module
from types import SimpleNamespace

from vesmod.EdgeMod import PowerCalculationResult, SpectrumEnsemble
from vesmod.cli.batch_policy import exit_code
from vesmod.cli.input_selection import select_input_files


def _load_exclusion_policy(spec):
    """Load a user-supplied Python exclusion function."""
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("Use MODULE:FUNCTION for --exclusion-policy.")
    policy = getattr(import_module(module), name)
    if not callable(policy):
        raise TypeError(f"Exclusion policy {spec} must be callable.")
    return policy


def _read_spectrum(path):
    """Load and validate the measured powers and metadata in a spectrum export."""
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict) or not {"modes", "powers", "r0"} <= data.keys():
        raise ValueError("Expected a per-vesicle spectrum JSON with modes, powers, and r0.")
    result = PowerCalculationResult(
        modes=data["modes"],
        powers=data["powers"],
        reference_radius=data["r0"],
        radius_unit="um",
        power_definition=data.get("power_definition"),
        power_unit=data.get("power_unit"),
    )
    if result.power_unit != "dimensionless":
        raise ValueError("Ensemble fitting requires dimensionless powers.")
    return SimpleNamespace(
        modes=result.modes, powers=result.powers, r0=result.reference_radius,
        power_definition=result.power_definition, power_unit=result.power_unit,
        kC=data.get("kC"), surface_tension=data.get("surface_tension"),
        fit_results=data.get("fit_results", []),
    )


def run_ensemble(args, config):
    """Apply the policy to saved spectra before aggregating and fitting."""
    paths, _ = select_input_files(args.input_path, ".json", args.recursive)
    paths = [
        path for path in paths
        if path.name != "edgemod_fit.json" and path.resolve() != args.output.resolve()
    ]
    if not paths:
        raise FileNotFoundError(f"No .json files found for {args.input_path}")
    policy = _load_exclusion_policy(args.exclusion_policy) if args.exclusion_policy else None
    ensemble = SpectrumEnsemble()
    replicas = []
    failed = 0
    power_definition = None
    report = {
        "exclusion_policy": args.exclusion_policy,
        "config": config.to_dict(),
        "weight_by_replica_sem": args.weight_by_replica_sem,
        "replicas": replicas,
        "fit": None,
    }

    with args.output.open("x", encoding="utf-8") as output:
        try:
            for path in paths:
                row = {"path": str(path.resolve())}
                replicas.append(row)
                try:
                    spectrum = _read_spectrum(path)
                    reason = policy(path, spectrum) if policy is not None else None
                    if reason is not None:
                        if not isinstance(reason, str) or not reason.strip():
                            raise ValueError("Exclusion policy must return None or a nonempty reason.")
                        row.update(status="excluded", reason=reason)
                        continue
                    if power_definition is not None and spectrum.power_definition != power_definition:
                        raise ValueError("Included spectra must share a power definition.")
                    ensemble.add_spectrum(spectrum.powers, spectrum.modes, spectrum.kC, r0=spectrum.r0)
                    power_definition = spectrum.power_definition
                    report["power_definition"] = power_definition
                    row.update(status="included", kC=spectrum.kC, radius_um=spectrum.r0)
                except (OSError, ValueError, TypeError, FloatingPointError) as exc:
                    row.update(status="error", reason=str(exc))
                    failed += 1
                    if args.error_policy == "fail-fast":
                        raise

            if not len(ensemble):
                raise ValueError("No vesicles available for the ensemble fit.")
            report["fit"] = ensemble.extract_kc_from_fit(
                config, weight_by_replica_sem=args.weight_by_replica_sem
            ).to_dict()
        except Exception as exc:
            report["error"] = str(exc)
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")

    return exit_code(failed, len(ensemble))
