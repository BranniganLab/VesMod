"""Internal frame-sequence names used during the frame-loading refactor."""

from .frame_source import (
    InMemoryFrameSequence,
    OnDemandFrameSequence,
    as_frame_source,
)


def as_frame_sequence(frames):
    """Normalize frame input using the concrete frame-sequence implementations."""
    return as_frame_source(frames)
