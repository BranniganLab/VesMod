"""Collision-safe mapping from selected inputs to output paths."""

from __future__ import annotations

from pathlib import Path


def relative_selected_path(path: Path, selector_root: Path) -> Path:
    """Return *path* relative to the root represented by a selector.

    An explicit file selector represents its parent directory for layout
    purposes, while directory and glob selectors represent the selected root.
    """
    resolved_path = path.expanduser().resolve()
    resolved_root = selector_root.expanduser().resolve()
    if resolved_path == resolved_root:
        return Path(resolved_path.name)
    try:
        return resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Selected path {resolved_path} is outside selector root {resolved_root}."
        ) from error


def map_output_path(
    path: Path,
    selector_root: Path,
    output_root: Path,
    *,
    suffix: str | None = None,
    stem_suffix: str = "",
) -> Path:
    """Map a selected input beneath an output root and validate containment.

    ``suffix`` replaces the input suffix; ``stem_suffix`` is appended to the
    filename stem before its suffix.  No directories are created.
    """
    relative = relative_selected_path(path, selector_root)
    if stem_suffix:
        relative = relative.with_name(f"{relative.stem}{stem_suffix}{relative.suffix}")
    if suffix is not None:
        relative = relative.with_suffix(suffix)
    root = output_root.expanduser().resolve()
    mapped = (root / relative).resolve()
    if not mapped.is_relative_to(root):
        raise ValueError(f"Mapped output path escapes output root: {mapped}")
    return mapped
