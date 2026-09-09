"""Shared policy and reporting for multi-input CLI operations."""

from __future__ import annotations

import argparse
import sys


def add_batch_policy_argument(parser: argparse.ArgumentParser) -> None:
    """Add the explicit fail-fast/keep-going policy option."""
    parser.add_argument(
        "--error-policy",
        choices=("keep-going", "fail-fast"),
        default="keep-going",
        help=(
            "How to handle per-input failures. keep-going reports each failure "
            "and processes the remaining inputs (default); fail-fast stops at "
            "the first failure."
        ),
    )


def exit_code(failed: int, succeeded: int) -> int:
    """Return 0 for success, 1 for partial failure, 2 for total failure."""
    if failed == 0:
        return 0
    return 2 if succeeded == 0 else 1


def report_batch_summary(
    processed: int,
    succeeded: int,
    skipped: int,
    failed: int,
) -> None:
    """Print concise, always-visible batch results to stderr."""
    print(
        "Batch summary: "
        f"processed={processed} succeeded={succeeded} "
        f"skipped={skipped} failed={failed}",
        file=sys.stderr,
    )
