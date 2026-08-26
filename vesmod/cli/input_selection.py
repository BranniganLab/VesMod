"""Shared CLI discovery for explicit, directory, and globbed inputs."""

from __future__ import annotations

import glob
import os
from pathlib import Path


def select_input_files(
    input_paths: list[Path],
    suffix: str,
    recursive: bool,
) -> tuple[list[Path], Path]:
    """Resolve CLI selectors to unique files and a stable relative-path root.

    ``input_paths`` may contain explicit files, directories, shell-expanded
    filenames, or unexpanded glob patterns. With ``recursive=True``, a glob
    pattern is also matched below subdirectories of its selected parent, so a
    selector such as ``pattern*.npz`` finds matching files recursively.
    """
    if not input_paths:
        raise ValueError("At least one input path is required.")

    expected_suffix = suffix.lower()
    matches: set[Path] = set()
    roots: list[Path] = []
    single_explicit_file: Path | None = None

    for selector in input_paths:
        selector_text = os.path.expanduser(str(selector))
        if glob.has_magic(selector_text):
            roots.append(_glob_root(selector_text))
            pattern = _recursive_glob(selector_text) if recursive else selector_text
            for match in glob.glob(pattern, recursive=recursive):
                candidate = Path(match).expanduser().resolve()
                if candidate.is_file() and candidate.suffix.lower() == expected_suffix:
                    matches.add(candidate)
            continue

        resolved = Path(selector_text).resolve()
        if resolved.is_file():
            if resolved.suffix.lower() != expected_suffix:
                raise ValueError(f"Expected a {expected_suffix} file, got: {resolved}")
            matches.add(resolved)
            roots.append(resolved.parent)
            if len(input_paths) == 1:
                single_explicit_file = resolved
            continue

        if resolved.is_dir():
            roots.append(resolved)
            candidates = resolved.rglob("*") if recursive else resolved.glob("*")
            matches.update(
                candidate.resolve()
                for candidate in candidates
                if candidate.is_file()
                and candidate.suffix.lower() == expected_suffix
            )
            continue

        raise FileNotFoundError(f"Input path or pattern does not exist: {selector}")

    if single_explicit_file is not None:
        root = single_explicit_file
    else:
        root = Path(os.path.commonpath([str(path) for path in roots])).resolve()
    return sorted(matches), root


def _recursive_glob(pattern: str) -> str:
    """Insert a recursive directory match before a glob's filename component."""
    path = Path(pattern)
    if "**" in path.parts:
        return pattern
    return str(path.parent / "**" / path.name)


def _glob_root(pattern: str) -> Path:
    """Return the non-glob prefix used as the relative-path root."""
    path = Path(pattern)
    parts = path.parts
    prefix: list[str] = []
    for part in parts:
        if glob.has_magic(part):
            break
        prefix.append(part)

    if not prefix:
        return Path.cwd().resolve()
    root = Path(*prefix)
    if path.is_absolute() and not root.is_absolute():
        root = Path(path.anchor) / root
    if root.suffix:
        root = root.parent
    return root.expanduser().resolve()
