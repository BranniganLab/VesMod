"""Path formatting for command-line messages."""

from pathlib import Path


def _display_path(path: Path) -> str:
    """Return a stable absolute path for command-line messages."""
    return str(path.expanduser().resolve())
