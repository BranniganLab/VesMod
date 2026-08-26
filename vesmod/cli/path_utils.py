"""Shared path helpers for VesMod command-line interfaces."""

from pathlib import Path


def _display_path(path: Path) -> str:
    """Return a stable absolute path for command-line messages."""
    return str(path.expanduser().resolve())


def _relative_input_path(path: Path, input_path: Path) -> Path:
    """Return one selected file relative to the user-selected input root."""
    resolved_path = path.expanduser().resolve()
    resolved_input = input_path.expanduser().resolve()
    if resolved_path == resolved_input:
        return Path(resolved_path.name)
    return resolved_path.relative_to(resolved_input)
