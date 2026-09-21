"""Experimental VesEdge APIs subject to change during method evaluation."""

from .internal_vesicle_qc import InternalVesicleQCConfig, check_internal_vesicle_selection

__all__ = [
    "InternalStructureConfig",
    "InternalHoughCircle",
    "InternalStructureFrameResult",
    "InternalStructureRegion",
    "InternalStructureVideoSummary",
    "InternalVesicleQCConfig",
    "check_internal_vesicle_selection",
    "detect_internal_structures",
    "select_temporally_supported_hough_circles",
    "summarize_internal_structures",
]


def __getattr__(name: str):
    """Load optional internal-structure dependencies only when requested."""
    if name not in {
        "InternalStructureConfig",
        "InternalHoughCircle",
        "InternalStructureFrameResult",
        "InternalStructureRegion",
        "InternalStructureVideoSummary",
        "detect_internal_structures",
        "select_temporally_supported_hough_circles",
        "summarize_internal_structures",
    }:
        raise AttributeError(name)
    from . import internal_structures

    return getattr(internal_structures, name)
