"""Fit an ensemble and record the selection of replica spectra."""

import json
from importlib import import_module

from vesmod.edgemod import EnsembleMember, SpectrumEnsemble
from vesmod.cli.batch_policy import exit_code
from vesmod.io.input_selection import select_input_files


def _load_exclusion_policy(spec):
    """Load a user-supplied Python exclusion function.

    Parameters
    ----------
    spec : str
        Importable policy in MODULE:FUNCTION form.

    Returns
    -------
    callable
        Policy taking a path and spectrum record and returning a reason or None.
    """
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("Use MODULE:FUNCTION for --exclusion-policy.")
    policy = getattr(import_module(module), name)
    if not callable(policy):
        raise TypeError(f"Exclusion policy {spec} must be callable.")
    return policy


def run_ensemble(args, config):
    """Apply the policy to saved spectra before aggregating and fitting.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ensemble CLI arguments.
    config : SpectrumFitConfig
        Physical settings for fitting the mean powers.

    Returns
    -------
    int
        Batch exit status.
    """
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
                    spectrum = EnsembleMember.from_json(path)
                    try:
                        reason = policy(path, spectrum) if policy is not None else None
                    except Exception as exc:
                        row.update(status="error", reason=str(exc))
                        if args.error_policy == "fail-fast":
                            raise
                        failed += 1
                        continue
                    if reason is not None:
                        if not isinstance(reason, str) or not reason.strip():
                            raise ValueError("Exclusion policy must return None or a nonempty reason.")
                        row.update(status="excluded", reason=reason)
                        continue
                    ensemble.add_spectrum(spectrum)
                    report["power_definition"] = spectrum.power_definition
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
