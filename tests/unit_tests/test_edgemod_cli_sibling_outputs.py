"""Regression coverage for EdgeMod selector-root overlap validation."""

from argparse import Namespace

from vesmod.cli import edgemod_cli


def _fit_args(input_paths, output_dir):
    """Return the minimal arguments needed by the stable fit batch."""
    return Namespace(
        command="fit",
        input_path=input_paths,
        recursive=True,
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


def test_external_fit_allows_sibling_output_for_sibling_input_directories(
    tmp_path,
    monkeypatch,
):
    """Test the common selector root does not make a sibling output overlap."""
    first_input = tmp_path / "condition_a"
    second_input = tmp_path / "condition_b"
    output_dir = tmp_path / "fits"
    first_input.mkdir()
    second_input.mkdir()
    (first_input / "first.npy").touch()
    (second_input / "second.npy").touch()

    args = _fit_args([first_input, second_input], output_dir)
    fit = Namespace(kC=12.5, surface_tension=1.0e-8)
    monkeypatch.setattr(edgemod_cli, "process_file", lambda path, parsed_args: fit)

    edgemod_cli._run_fit(args)

    assert (output_dir / "edgemod_fit.json").is_file()
    assert (output_dir / "fit_summary.csv").is_file()
