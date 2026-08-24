"""Tests for EdgeMod command-line fit configuration."""

from argparse import Namespace
from pathlib import Path
import json

import numpy as np
import pytest

from vesmod.EdgeMod import FixedFitRangeSelector, QMinusThreeFitRangeSelector
from vesmod.cli import edgemod_cli
from vesmod.cli.edgemod_cli import build_fit_config, output_path_for, process_file


def _args(**overrides):
    """Return standard parsed CLI arguments with optional overrides."""
    values = {
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


def test_build_fit_config_uses_fixed_selector_by_default():
    """Test the CLI preserves the historical fixed-range behavior."""
    config = build_fit_config(_args())

    assert isinstance(config.range_selector, FixedFitRangeSelector)
    assert config.range_selector.lower_bound == 3
    assert config.range_selector.upper_bound == 8
    assert config.lmax == 500
    assert config.free_sigma is True
    assert config.temperature == 295.0


def test_build_fit_config_constructs_dynamic_selector():
    """Test all dynamic-selection arguments are propagated into the config."""
    config = build_fit_config(
        _args(
            dynamic_range=True,
            upper_fitting_bound=15,
            min_modes=5,
            slope_tolerance=0.2,
            max_log_rmse=0.1,
            fixed_sigma=True,
        )
    )

    assert isinstance(config.range_selector, QMinusThreeFitRangeSelector)
    assert config.range_selector.lower_bound == 3
    assert config.range_selector.upper_bound == 15
    assert config.range_selector.min_modes == 5
    assert config.range_selector.slope_tolerance == 0.2
    assert config.range_selector.max_log_rmse == 0.1
    assert config.free_sigma is False


def test_build_fit_config_requires_explicit_dynamic_thresholds():
    """Test dynamic fitting cannot silently use empirical acceptance defaults."""
    with pytest.raises(ValueError, match="--slope-tolerance"):
        build_fit_config(
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
    """Test rejected dynamic fits write diagnostics before raising."""
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
    assert data["fit_range_selection"]["accepted"] is False
    assert data["fit_range_selection"]["reason"] is not None
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
    args = _args()
    args.input_path = tmp_path
    args.recursive = True
    processed = []

    monkeypatch.setattr(edgemod_cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        edgemod_cli,
        "iter_npy_files",
        lambda input_path, recursive: [failed_path, successful_path],
    )

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
    args = _args()
    args.input_path = failed_path
    args.recursive = False

    monkeypatch.setattr(edgemod_cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        edgemod_cli,
        "iter_npy_files",
        lambda input_path, recursive: [failed_path],
    )
    monkeypatch.setattr(
        edgemod_cli,
        "process_file",
        lambda path, parsed_args: (_ for _ in ()).throw(ValueError("fit failed")),
    )

    with pytest.raises(ValueError, match="fit failed"):
        edgemod_cli.main()
