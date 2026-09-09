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
