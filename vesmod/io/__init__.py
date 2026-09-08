"""Shared I/O infrastructure without domain-specific file schemas."""

from .checkpoint_sources import build_video_filename_index, open_checkpoint_frames, resolve_source_path
from .path_mapping import map_output_path, relative_selected_path

__all__ = [
    "build_video_filename_index",
    "map_output_path",
    "open_checkpoint_frames",
    "relative_selected_path",
    "resolve_source_path",
]
