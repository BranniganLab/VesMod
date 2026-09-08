"""Locate checkpoint source videos and open their frames lazily.

This module deliberately knows no checkpoint schema, QC configuration, or CLI
arguments. Domain code supplies the recorded source path and any search roots.
"""

from __future__ import annotations

from pathlib import Path

from vesmod.VesEdge import FrameSource, open_frame_source


def build_video_filename_index(checkpoint_paths: list[Path], video_root: Path | None) -> dict[str, tuple[Path, ...]]:
    """Index possible source videos once for a batch."""
    roots = {path.expanduser().resolve().parent for path in checkpoint_paths}
    if video_root is not None and video_root.expanduser().resolve().is_dir():
        roots.add(video_root.expanduser().resolve())
    matches: dict[str, set[Path]] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                matches.setdefault(path.name.lower(), set()).add(path.resolve())
    return {name: tuple(sorted(paths)) for name, paths in matches.items()}


def resolve_source_path(stored_path: str | Path | None, checkpoint_path: Path, video_root: Path | None = None, video_index: dict[str, tuple[Path, ...]] | None = None) -> Path:
    """Resolve a recorded source path, then unambiguous filename fallbacks."""
    checkpoint = checkpoint_path.expanduser().resolve()
    if stored_path is not None:
        stored = Path(stored_path).expanduser()
        candidates = [stored]
        if not stored.is_absolute():
            candidates.append(checkpoint.parent / stored)
        candidates.append(checkpoint.parent / stored.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        name = stored.name
    else:
        name = checkpoint.with_suffix(".nd2").name
    roots = [checkpoint.parent]
    if video_root is not None:
        root = video_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Video root does not exist or is not a directory: {root}")
        if root not in roots:
            roots.append(root)
    matches = ([path for path in video_index.get(name.lower(), ()) if any(path.is_relative_to(root) for root in roots)] if video_index is not None else sorted({path.resolve() for root in roots for path in root.rglob("*") if path.is_file() and path.name.lower() == name.lower()}))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple source videos match {name}: {', '.join(map(str, matches))}")
    if stored_path is None:
        raise FileNotFoundError(f"Checkpoint does not record a source video path and no matching {name} was found beside it or under --video-root.")
    raise FileNotFoundError(f"Source video does not exist: {stored_path}. No matching {name} was found beside the checkpoint or under --video-root.")


def open_checkpoint_frames(source_path: Path) -> FrameSource:
    """Open source frames lazily for a checkpoint-backed workflow."""
    return open_frame_source(source_path)
