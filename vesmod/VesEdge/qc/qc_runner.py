"""Execute registered VesEdge quality-control checks."""

from __future__ import annotations

from numpy.typing import NDArray

from ..io.frame_source import OnDemandFrameSequence
from ..models import EdgeDetection, VesicleQCResult
from . import qc_registry
from .qc_check import QCCheck
from .qc_config import VesicleQCConfig


Frames = OnDemandFrameSequence | NDArray | None


def checks_requiring_source_video_frames(config: VesicleQCConfig) -> tuple[QCCheck, ...]:
    """Return enabled checks that need source images."""
    return tuple(
        check for check in qc_registry.QC_CHECKS
        if check.requires_source_video_frames and check.enabled(config.get_config(check.name))
    )


def run_configured_qc_checks(
    detections: list[EdgeDetection],
    config: VesicleQCConfig,
    frames: Frames = None,
) -> VesicleQCResult:
    """Run the explicit ordered registry and combine aggregate outcomes."""
    results = {}
    for check in qc_registry.QC_CHECKS:
        check_config = config.get_config(check.name)
        if not check.enabled(check_config):
            continue
        if check.requires_source_video_frames and frames is None:
            raise ValueError(
                f"{check.name.replace('_', '-').capitalize()} QC is enabled "
                "but source video frames were not supplied."
            )
        result = check.run(detections, check_config, frames)
        check.validate_result(result)
        if result.frame_indices != tuple(edge.frame_index for edge in detections):
            raise ValueError(f"{check.name} must return decisions for every supplied source frame.")
        for index, detection in enumerate(detections):
            if result.rejected[index]:
                detection.qc.flags.add(check.name)
            else:
                detection.qc.flags.discard(check.name)
            detection.qc.diagnostics.update({
                name: values[index] for name, values in result.diagnostics.items()
            })
        results[check.name] = result
    return VesicleQCResult(
        config=config, results=results,
        frame_indices=tuple(edge.frame_index for edge in detections),
    )
