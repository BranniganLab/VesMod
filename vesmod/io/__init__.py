"""Shared I/O infrastructure without domain-specific file schemas."""

from .checkpoint_sources import build_video_filename_index, open_checkpoint_frames, resolve_source_path

__all__ = ["build_video_filename_index", "open_checkpoint_frames", "resolve_source_path"]
