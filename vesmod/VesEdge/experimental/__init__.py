"""Experimental VesEdge APIs subject to change during method evaluation."""

from .internal_structures import (
    InternalStructureConfig,
    InternalStructureFrameResult,
    InternalStructureRegion,
    InternalStructureVideoSummary,
    detect_internal_structures,
    summarize_internal_structures,
)

__all__ = [
    "InternalStructureConfig",
    "InternalStructureFrameResult",
    "InternalStructureRegion",
    "InternalStructureVideoSummary",
    "detect_internal_structures",
    "summarize_internal_structures",
]
