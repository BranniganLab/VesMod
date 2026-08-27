"""Tests for EdgeMod command-line fit configuration."""

from argparse import Namespace
from pathlib import Path
import json
import sys

import numpy as np
import pytest

from vesmod.EdgeMod import SpectrumFitConfig
from vesmod.EdgeMod.experimental import QMinusThreeRangeSelector
from vesmod.EdgeMod.experimental import TemporalRMSConfig
from vesmod.cli import edgemod_cli
from vesmod.cli.edgemod_cli import (
    build_dynamic_selector,
    build_fit_config,
    output_path_for,
    process_file,
)


def _args(**overrides):
    """Return standard parsed CLI arguments with optional overrides."""
    values = {
        "command": "fit",
        "dynamic_range": False,
        "lower_fitting_bound": 3,
        "upper_fitting_bound": 8,
        "min_modes": None,
        "slope_tolerance": None,
        "max_log_rmse": None,
        "lmax": 500,
        "fixed_sigma": False,
        "temperature": 295.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_parse_args_preserves_direct_fit_command(monkeypatch):
    """Test the established command directly invokes core physical fitting."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["edgemod", "edges", "--fixed-sigma"],
    )

    args = edgemod_cli.parse_args()

    assert args.command == "fit"
    assert args.input_path == Path("edges")
    assert args.fixed_sigma


def test_parse_args_selects_temporal_rms_subcommand(monkeypatch, tmp_path):
    """Test experimental screening options are scoped to their own stage."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edgemod",
            "experimental",
            "temporal-rms",
            "edges",
            "--output-dir",
            str(tmp_path),
            "--cutoff-nm",
            "50",
        ],
    )

    args = edgemod_cli.parse_args()

    assert args.command == "temporal-rms"
    assert args.output_dir == tmp_path
    assert args.cutoff_nm == pytest.approx(50.0)


def test_build_fit_config_uses_fixed_bounds_by_default():
    """Test the core CLI config preserves historical fixed-range behavior."""
    config = build_fit_config(_args())

    assert isinstance(config, SpectrumFitConfig)
    assert config.lower_bound == 3
    assert config.upper_bound == 8
    assert config.lmax == 500
    assert config.free_sigma is True
    assert config.temperature == 295.0


def test_build_dynamic_selector_is_separate_from_core_config():
    """Test experimental selector construction is an explicit CLI step."""
    args = _args(
        dynamic_range=True,
        upper_fitting_bound=15,
        min_modes=5,
        slope_tolerance=0.2,
        max_log_rmse=0.1,
        fixed_sigma=True,
    )

    config = build_fit_config(args)
    selector = build_dynamic_selector(args)

    assert isinstance(selector, QMinusThreeRangeSelector)
    assert selector.lower_bound == 3
    assert selector.upper_bound == 15
    assert selector.min_modes == 5
    assert selector.slope_tolerance == 0.2
    assert selector.max_log_rmse == 0.1
    assert config.lower_bound == 3
    assert config.upper_bound == 15
    assert config.free_sigma is False
    assert not hasattr(config, "range_selector")


def test_build_dynamic_selector_requires_explicit_thresholds():
    """Test experimental selection cannot silently use empirical defaults."""
    with pytest.raises(ValueError, match="--slope-tolerance"):
        build_dynamic_selector(
            _args(
                dynamic_range=True,
                min_modes=5,
                max_log_rmse=0.1,
            )
        )


def test_dynamic_output_path_does_not_overwrite_fixed_output():
    """Test fixed and dynamic CLI runs use distinct JSON filenames."""
    path = Path("sample.npy")

    assert output_path_for(path, dynamic_range=False) == Path("sample.json")
    assert output_path_for(path, dynamic_range=True) == Path("sample.dynamic.json")


def test_process_file_serializes_dynamic_rejection_diagnostics(tmp_path):
    """Test rejected experimental selection writes diagnostics before raising."""
    path = tmp_path / "sample.npy"
    np.save(path, np.ones((3, 12), dtype=float))
    args = _args(
        dynamic_range=True,
        upper_fitting_bound=12,
        min_modes=5,
        slope_tolerance=0.1,
        max_log_rmse=0.05,
    )

    with pytest.raises(ValueError):
        process_file(path, args)

    output_path = tmp_path / "sample.dynamic.json"
    assert output_path.is_file()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    diagnostics = data["experimental"]["dynamic_range_selection"]
    assert diagnostics["accepted"] is False
    assert diagnostics["reason"] is not None
    assert data["kC"] is None
    assert data["surface_tension"] is None


@pytest.mark.parametrize(
    "error", [ValueError("fit failed"), FloatingPointError("fit failed")]
)
def test_recursive_run_skips_failed_fit_and_continues(
    monkeypatch, capsys, tmp_path, error
):
    """Test one failed spectrum does not abort a recursive batch."""
    failed_path = tmp_path / "failed.npy"
    successful_path = tmp_path / "successful.npy"
    failed_path.touch()
    successful_path.touch()
    args = _args()
    args.input_path = tmp_path
    args.recursive = True
    processed = []

    monkeypatch.setattr(edgemod_cli, "parse_args", lambda: args)
    def fake_process_file(path, parsed_args):
        processed.append(path)
        if path == failed_path:
            raise error

    monkeypatch.setattr(edgemod_cli, "process_file", fake_process_file)

    edgemod_cli.main()

    assert processed == [failed_path, successful_path]
    assert f"Skipping {failed_path}: fit failed" in capsys.readouterr().err


def test_nonrecursive_run_propagates_failed_fit(monkeypatch, tmp_path):
    """Test a direct single-spectrum run still reports failure to the caller."""
    failed_path = tmp_path / "failed.npy"
    failed_path.touch()
    args = _args()
    args.input_path = failed_path
    args.recursive = False

    monkeypatch.setattr(edgemod_cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        edgemod_cli,
        "process_file",
        lambda path, parsed_args: (_ for _ in ()).throw(ValueError("fit failed")),
    )

    with pytest.raises(ValueError, match="fit failed"):
        edgemod_cli.main()


def test_process_temporal_rms_file_exports_only_included_input(tmp_path):
    """Test the experimental stage writes only trajectories passing its cutoff."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    static_path = input_dir / "static.npy"
    np.save(static_path, np.ones((10, 120), dtype=float))
    args = Namespace(
        input_path=input_dir,
        output_dir=output_dir,
        overwrite=False,
    )
    config = TemporalRMSConfig(cutoff_nm=50.0)

    row, result = edgemod_cli.process_temporal_rms_file(
        static_path,
        args,
        config,
    )

    assert result is not None
    assert result.included is False
    assert row["status"] == "below_cutoff"
    assert not (output_dir / "static.npy").exists()


def test_run_temporal_rms_writes_vesedge_style_batch_outputs(tmp_path):
    """Test screening writes provenance, summary, histogram, and accepted arrays."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    angles = 2.0 * np.pi * np.arange(120) / 120
    phases = 2.0 * np.pi * np.arange(20) / 20
    radii = np.array(
        [10.0 + 0.1 * np.cos(3.0 * angles + phase) for phase in phases]
    )
    np.save(input_dir / "moving.npy", radii)
    args = Namespace(
        input_path=input_dir,
        output_dir=output_dir,
        recursive=False,
        lower_bound=3,
        upper_bound=8,
        cutoff_nm=50.0,
        overwrite=False,
    )

    edgemod_cli._run_temporal_rms(args)

    assert (output_dir / "moving.npy").is_file()
    assert (output_dir / "temporal_rms_qc.json").is_file()
    assert (output_dir / "temporal_rms_summary.csv").is_file()
    assert (output_dir / "temporal_rms_histogram.png").is_file()


def test_temporal_rms_rejects_equal_input_and_output_without_deleting_source(
    tmp_path,
):
    """Test an in-place batch is rejected before source arrays are touched."""
    source_path = tmp_path / "source.npy"
    np.save(source_path, np.ones((10, 120), dtype=float))
    args = Namespace(
        input_path=tmp_path,
        output_dir=tmp_path,
        recursive=False,
        lower_bound=3,
        upper_bound=8,
        cutoff_nm=50.0,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="must not overlap"):
        edgemod_cli._run_temporal_rms(args)

    assert source_path.is_file()


def test_remove_temporal_rms_artifacts_uses_validated_export_manifest(tmp_path):
    """Test cleanup removes recorded exports while preserving unrelated arrays."""
    exported_path = tmp_path / "accepted.npy"
    unrelated_path = tmp_path / "unrelated.npy"
    np.save(exported_path, np.ones((2, 2), dtype=float))
    np.save(unrelated_path, np.ones((2, 2), dtype=float))
    provenance = {
        "experimental_method": "temporal_rms",
        "exported_files": ["accepted.npy"],
    }

    edgemod_cli._remove_temporal_rms_artifacts(tmp_path, provenance)

    assert not exported_path.exists()
    assert unrelated_path.is_file()



def test_parse_args_accepts_external_fit_output(monkeypatch, tmp_path):
    """Test stable fitting exposes output and overwrite options."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "edgemod",
            "edges",
            "--output-dir",
            str(tmp_path),
            "--overwrite",
        ],
    )

    args = edgemod_cli.parse_args()

    assert args.output_dir == tmp_path
    assert args.overwrite


def test_fit_output_path_preserves_relative_directories(tmp_path):
    """Test external fit outputs mirror the selected input hierarchy."""
    input_dir = tmp_path / "input"
    path = input_dir / "condition" / "sample.npy"
    path.parent.mkdir(parents=True)
    path.touch()
    args = _args()
    args.input_path = input_dir
    args.output_dir = tmp_path / "output"

    output_path = edgemod_cli._fit_output_path(path, args)

    assert output_path == tmp_path / "output" / "condition" / "sample.json"
    assert output_path.parent.is_dir()


def test_fit_output_path_keeps_legacy_beside_input_behavior(tmp_path):
    """Test omitting output-dir leaves the established path unchanged."""
    path = tmp_path / "sample.npy"
    args = _args()

    assert edgemod_cli._fit_output_path(path, args) == tmp_path / "sample.json"


def test_external_fit_batch_writes_provenance_and_summary(
    monkeypatch,
    tmp_path,
):
    """Test external batches record every successfully attempted input."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source_path = input_dir / "sample.npy"
    source_path.touch()
    args = _args()
    args.input_path = input_dir
    args.output_dir = output_dir
    args.overwrite = False
    args.recursive = True
    fit = Namespace(kC=12.5, surface_tension=1.0e-8)

    monkeypatch.setattr(edgemod_cli, "process_file", lambda path, args: fit)

    edgemod_cli._run_fit(args)

    provenance = json.loads(
        (output_dir / "edgemod_fit.json").read_text(encoding="utf-8")
    )
    summary = (output_dir / "fit_summary.csv").read_text(encoding="utf-8")
    assert provenance["analysis"] == "edgemod_fit"
    assert provenance["input_manifest"] == [str(source_path.resolve())]
    assert "sample.npy,ok,12.5,1e-08," in summary


def test_external_fit_batch_summarizes_recursive_failure(
    monkeypatch,
    tmp_path,
):
    """Test recursive fitting failures are retained in the batch summary."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source_path = input_dir / "sample.npy"
    source_path.touch()
    args = _args()
    args.input_path = input_dir
    args.output_dir = output_dir
    args.overwrite = False
    args.recursive = True

    monkeypatch.setattr(
        edgemod_cli,
        "process_file",
        lambda path, args: (_ for _ in ()).throw(ValueError("fit failed")),
    )

    edgemod_cli._run_fit(args)

    summary = (output_dir / "fit_summary.csv").read_text(encoding="utf-8")
    assert "sample.npy,fit_error,,,fit failed" in summary


def test_external_fit_rejects_overlapping_paths(tmp_path):
    """Test recursive outputs cannot be placed inside the input tree."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.npy").touch()
    args = _args()
    args.input_path = input_dir
    args.output_dir = input_dir / "fits"
    args.overwrite = False
    args.recursive = True

    with pytest.raises(ValueError, match="must not overlap"):
        edgemod_cli._run_fit(args)


def test_incompatible_fit_provenance_requires_overwrite(tmp_path):
    """Test an external directory cannot silently mix fit configurations."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source_path = input_dir / "sample.npy"
    source_path.touch()
    args = _args()
    args.input_path = input_dir
    args.output_dir = output_dir
    args.overwrite = False
    args.recursive = True

    edgemod_cli._prepare_fit_output(args, [source_path])
    args.temperature = 300.0

    with pytest.raises(ValueError, match="different input selection"):
        edgemod_cli._prepare_fit_output(args, [source_path])


def test_remove_fit_artifacts_preserves_unrelated_files(tmp_path):
    """Test overwrite cleanup removes only manifest-recorded fit artifacts."""
    managed = tmp_path / "condition" / "sample.json"
    unrelated = tmp_path / "notes.json"
    managed.parent.mkdir()
    managed.write_text("{}", encoding="utf-8")
    unrelated.write_text("{}", encoding="utf-8")
    provenance = {
        "analysis": "edgemod_fit",
        "managed_artifacts": ["condition/sample.json"],
    }

    edgemod_cli._remove_fit_artifacts(tmp_path, provenance)

    assert not managed.exists()
    assert unrelated.is_file()
