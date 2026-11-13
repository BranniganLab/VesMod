"""Fit an ensemble and record the selection of replica spectra."""

import json
from importlib import import_module

from vesmod.EdgeMod import Spectrum, SpectrumEnsemble
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


def run_ensemble(args, config):
    """Apply the policy before fitting and aggregating each accepted replica."""
    paths, _ = select_input_files(args.input_path, ".npy", args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .npy files found for {args.input_path}")
    policy = _load_exclusion_policy(args.exclusion_policy) if args.exclusion_policy else None
    ensemble = SpectrumEnsemble()
    replicas = []
    failed = 0
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
                    spectrum = Spectrum(path)
                    reason = policy(path, spectrum) if policy is not None else None
                    if reason is not None:
                        if not isinstance(reason, str) or not reason.strip():
                            raise ValueError("Exclusion policy must return None or a nonempty reason.")
                        row.update(status="excluded", reason=reason)
                        continue
                    fit = spectrum.extract_kc_from_fit(config)
                    ensemble.add_spectrum(spectrum.powers, spectrum.modes, fit.kC, r0=spectrum.r0)
                    row.update(status="included", kC=fit.kC, radius_um=spectrum.r0)
                except (OSError, ValueError, FloatingPointError) as exc:
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
