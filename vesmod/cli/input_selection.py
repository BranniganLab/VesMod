"""Shared CLI discovery for explicit, directory, and globbed inputs."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path


class InputPathsAction(argparse.Action):
    """Preserve one-input Path behavior while accepting multiple selectors."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        """Store one Path directly, or multiple Paths as a list."""
        del parser, option_string
        setattr(namespace, self.dest, values[0] if len(values) == 1 else values)


def normalize_input_paths(input_path: Path | list[Path]) -> list[Path]:
    """Return one or more parsed selectors as a list."""
    return input_path if isinstance(input_path, list) else [input_path]


def select_input_files(
    input_path: Path | list[Path],
    suffix: str,
    recursive: bool,
    *,
    return_roots: bool = False,
) -> tuple[list[Path], Path] | tuple[list[Path], Path, tuple[Path, ...]]:
    """Resolve CLI selectors to unique files and a stable relative-path root.

    Selectors may be explicit files, directories, shell-expanded filenames, or
    unexpanded glob patterns. With ``recursive=True``, a glob pattern is also
    matched below subdirectories of its selected parent, so a selector such as
    ``pattern*.npz`` finds matching files recursively.

    Parameters
    ----------
    input_path : Path or list[Path]
        One or more user-supplied selectors.
    suffix : str
        Required file suffix, including the leading period.
    recursive : bool
        Whether directory and glob selectors should search subdirectories.
    return_roots : bool, optional
        When True, also return the resolved root represented by each original
        selector. Explicit files remain file roots, while directory and glob
        selectors retain their selected directory roots. This keeps overlap
        validation independent of the synthetic common root used only for
        relative output paths.

    Returns
    -------
    tuple
        Selected files and their common relative-path root. When
        ``return_roots`` is True, a tuple of unique resolved selector roots is
        returned as the third item.
    """
    input_paths = normalize_input_paths(input_path)
    expected_suffix = suffix.lower()
    matches: set[Path] = set()
    roots: list[Path] = []
    selector_roots: list[Path] = []
    single_explicit_file: Path | None = None

    for selector in input_paths:
        selector_text = os.path.expanduser(str(selector))
        if glob.has_magic(selector_text):
            glob_root = _glob_root(selector_text)
            roots.append(glob_root)
            selector_roots.append(glob_root)
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
            selector_roots.append(resolved)
            if len(input_paths) == 1:
                single_explicit_file = resolved
            continue

        if resolved.is_dir():
            roots.append(resolved)
            selector_roots.append(resolved)
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

    selected = sorted(matches)
    if return_roots:
        unique_roots = tuple(
            dict.fromkeys(path.resolve() for path in selector_roots)
        )
        return selected, root, unique_roots
    return selected, root


def _recursive_glob(pattern: str) -> str:
    """Insert a recursive directory match before a glob's filename component."""
    path = Path(pattern)
    if "**" in path.parts:
        return pattern
    return str(path.parent / "**" / path.name)


def _glob_root(pattern: str) -> Path:
    """Return the non-glob prefix used as the relative-path root."""
    path = Path(pattern)
    prefix: list[str] = []
    for part in path.parts:
        if glob.has_magic(part):
            break
        prefix.append(part)

    if not prefix:
        return Path.cwd().resolve()
    root = Path(*prefix)
    if root.suffix:
        root = root.parent
    return root.expanduser().resolve()
