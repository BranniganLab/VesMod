import argparse

from vesmod.cli.batch_policy import add_batch_policy_argument, exit_code


def test_exit_codes_distinguish_success_partial_and_total_failure():
    assert exit_code(0, 3) == 0
    assert exit_code(1, 2) == 1
    assert exit_code(3, 0) == 2


def test_batch_policy_defaults_to_keep_going():
    parser = argparse.ArgumentParser()
    add_batch_policy_argument(parser)
    assert parser.parse_args([]).error_policy == "keep-going"
    assert parser.parse_args(["--error-policy", "fail-fast"]).error_policy == "fail-fast"


def test_extract_batch_counts_skips_separately(monkeypatch, capsys):
    from pathlib import Path
    from argparse import Namespace
    import vesmod.cli.vesedge_cli as vesedge_cli

    paths = [Path("one.nd2"), Path("two.nd2"), Path("three.nd2")]
    monkeypatch.setattr(
        vesedge_cli,
        "select_input_files",
        lambda *args, **kwargs: (paths, Path("inputs")),
    )
    monkeypatch.setattr(
        vesedge_cli,
        "process_extract_file",
        lambda path, args: {paths[0]: None, paths[1]: True, paths[2]: False}[path],
    )

    args = Namespace(
        input_path=Path("inputs"),
        recursive=False,
        error_policy="keep-going",
    )

    assert vesedge_cli._run_extract(args) == 1
    assert "processed=3 succeeded=1 skipped=1 failed=1" in capsys.readouterr().err


def test_gif_batch_classifies_skip_and_failure_separately(monkeypatch, capsys, tmp_path):
    from pathlib import Path
    from argparse import Namespace
    import vesmod.cli.gif_cli as gif_cli

    paths = [Path("one.npz"), Path("two.npz"), Path("three.npz")]
    monkeypatch.setattr(
        gif_cli,
        "select_input_files",
        lambda *args, **kwargs: (paths, Path("inputs")),
    )
    monkeypatch.setattr(
        gif_cli,
        "process_gif_file",
        lambda checkpoint, args, selection: {
            paths[0]: None,
            paths[1]: False,
            paths[2]: True,
        }[checkpoint],
    )

    args = Namespace(
        input_path=Path("inputs"),
        recursive=False,
        style="edges",
        qc_dir=None,
        output_dir=tmp_path,
        error_policy="keep-going",
    )

    assert gif_cli.run_gif(args) == 1
    assert "processed=3 succeeded=1 skipped=1 failed=1" in capsys.readouterr().err
