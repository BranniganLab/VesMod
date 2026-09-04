"""Keep important documented VesEdge commands aligned with the CLI parser."""

import shlex
import sys
from pathlib import Path

import pytest

from vesmod.cli import vesedge_cli


@pytest.mark.parametrize(
    "command",
    [
        (
            'vesedge extract "./videos" --pixels-per-micron 13.44 '
            "--downsample --n-samples 120 --output-dir ./checkpoints"
        ),
        (
            'vesedge gif "./checkpoints" --output-dir ./gifs/original '
            "--style original"
        ),
        (
            'vesedge gif "./checkpoints" --output-dir ./gifs/edges '
            "--style edges"
        ),
        (
            'vesedge gif "./checkpoints" --output-dir ./gifs/qc_standard '
            "--style qc --qc-dir ./results/qc_standard"
        ),
        (
            'vesedge qc "./checkpoints" --curvature-threshold 0.059 '
            "--max-relative-area-deviation 0.25 "
            "--output-dir ./results/qc_standard"
        ),
    ],
)
def test_recommended_vesedge_commands_parse(monkeypatch, command):
    """Ensure the recommended extraction, GIF, and QC examples stay valid."""
    monkeypatch.setattr(sys, "argv", shlex.split(command))

    vesedge_cli.parse_args()


def test_extraction_docs_do_not_advertise_removed_no_gif_option():
    """Ensure extraction documentation leaves GIF creation to `vesedge gif`."""
    readme = Path("README.md").read_text(encoding="utf-8")
    cli_guide = Path("docs/VesEdge_CLI_README.md").read_text(encoding="utf-8")
    extraction_guide = cli_guide.split("# `vesedge qc`", maxsplit=1)[0]

    assert "--no-gif" not in readme
    assert "--no-gif" not in extraction_guide
    assert "# `vesedge gif`" in cli_guide
