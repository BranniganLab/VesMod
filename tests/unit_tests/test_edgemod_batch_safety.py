"""Regression tests for EdgeMod external batch-output safety."""

from argparse import Namespace
import csv
import json

from vesmod.cli import edgemod_cli


def _fit_args(input_path, output_dir):
    """Return the minimum stable fit arguments used by external batches."""
    return Namespace(
        command="fit",
        input_path=input_path,
        recursive=False,
        dynamic_range=False,
        lower_fitting_bound=3,
        upper_fitting_bound=8,
        min_modes=None,
        slope_tolerance=None,
        max_log_rmse=None,
        lmax=500,
        fixed_sigma=False,
        temperature=295.0,
        output_dir=output_dir,
        overwrite=False,
    )


def test_fit_allows_sibling_output_for_sibling_input_directories(
    tmp_path,
    monkeypatch,
):
    """Test a common synthetic root does not falsely overlap sibling output."""
    first_dir = tmp_path / "condition_a"
    second_dir = tmp_path / "condition_b"
    output_dir = tmp_path / "fits"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / "first.npy"
    second_path = second_dir / "second.npy"
    first_path.touch()
    second_path.touch()
    args = _fit_args([first_dir, second_dir], output_dir)
    fit = Namespace(kC=12.0, surface_tension=1.0e-8)

    monkeypatch.setattr(edgemod_cli, "process_file", lambda path, parsed: fit)

    edgemod_cli._run_fit(args)

    assert (output_dir / "fit_summary.csv").is_file()
    provenance = json.loads(
        (output_dir / "edgemod_fit.json").read_text(encoding="utf-8")
    )
    assert provenance["input_manifest"] == [
        str(first_path.resolve()),
        str(second_path.resolve()),
    ]


def test_fit_summary_prefixes_formula_like_file_and_error_cells(tmp_path):
    """Test untrusted spreadsheet-formula-like text is escaped in CSV output."""
    output_dir = tmp_path / "fits"
    output_dir.mkdir()
    (output_dir / "edgemod_fit.json").write_text(
        json.dumps({"managed_artifacts": []}),
        encoding="utf-8",
    )
    args = Namespace(output_dir=output_dir, dynamic_range=False)
    rows = [
        {
            "file": "=formula.npy",
            "status": "fit_error",
            "kC": "",
            "surface_tension": "",
            "error": "+formula error",
        }
    ]

    edgemod_cli._write_fit_batch_outputs(args, rows)

    with (output_dir / "fit_summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["file"] == "'=formula.npy"
    assert row["error"] == "'+formula error"
    assert rows[0]["file"] == "=formula.npy"
    assert rows[0]["error"] == "+formula error"
