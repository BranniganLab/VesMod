"""Composable built-in VesEdge quality-control checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from numpy.typing import NDArray

from .area_qc import check_area_deviation
from .config import EdgeQCConfig
from .edge_filtering import check_curvature
from .experimental.internal_vesicle_qc import check_internal_vesicle_selection
from .frame_source import FrameSource
from .localized_deviation_qc import check_localized_deviation
from .minimum_radius_qc import check_minimum_radius
from .models import (
    AreaQCResult,
    CurvatureQCResult,
    EdgeDetection,
    QCFlag,
    TrajectoryQCFlag,
    InternalVesicleQCResult,
)


Frames = FrameSource | NDArray | None


@dataclass(frozen=True)
class QCCheckOutcome:
    """One check's optional aggregate result and trajectory flags."""

    curvature: CurvatureQCResult | None = None
    area: AreaQCResult | None = None
    internal_vesicle: InternalVesicleQCResult | None = None
    trajectory_flags: frozenset[TrajectoryQCFlag] = frozenset()


class QCCheck(Protocol):
    """Typed contract for a configured built-in quality-control check."""

    name: str
    requires_frames: bool

    def enabled(self, config: EdgeQCConfig) -> bool:
        """Return whether this check is enabled by configuration."""

    def run(
        self,
        detections: list[EdgeDetection],
        config: EdgeQCConfig,
        frames: Frames,
    ) -> QCCheckOutcome:
        """Apply the check and return any aggregate outcome."""


class CurvatureCheck:
    name = "curvature"
    requires_frames = False

    def enabled(self, config: EdgeQCConfig) -> bool:
        return config.curvature.enabled

    def run(self, detections, config, frames) -> QCCheckOutcome:
        del frames
        for detection in detections:
            check_curvature(detection, threshold=config.curvature.threshold)
        scores = tuple(
            float(detection.qc.curvature_score)
            if detection.qc.curvature_score is not None
            else float("nan")
            for detection in detections
        )
        return QCCheckOutcome(
            curvature=CurvatureQCResult(
                scores=scores,
                rejected_count=sum(
                    QCFlag.CURVATURE in detection.qc.flags
                    for detection in detections
                ),
            )
        )


class MinimumRadiusCheck:
    name = "minimum_radius"
    requires_frames = False

    def enabled(self, config: EdgeQCConfig) -> bool:
        return config.minimum_radius.enabled

    def run(self, detections, config, frames) -> QCCheckOutcome:
        del frames
        for detection in detections:
            check_minimum_radius(
                detection,
                min_median_radius_pixels=(
                    config.minimum_radius.min_median_radius_pixels
                ),
            )
        return QCCheckOutcome()


class LocalizedDeviationCheck:
    name = "localized_deviation"
    requires_frames = False

    def enabled(self, config: EdgeQCConfig) -> bool:
        return config.baseline.enabled

    def run(self, detections, config, frames) -> QCCheckOutcome:
        del frames
        for detection in detections:
            check_localized_deviation(
                detection,
                config.baseline.order,
                config.baseline.max_residual_fraction,
            )
        return QCCheckOutcome()


class AreaDeviationCheck:
    name = "area"
    requires_frames = False

    def enabled(self, config: EdgeQCConfig) -> bool:
        return config.area.enabled

    def run(self, detections, config, frames) -> QCCheckOutcome:
        del frames
        return QCCheckOutcome(
            area=check_area_deviation(
                detections,
                config.area.max_relative_deviation,
            )
        )


class InternalVesicleCheck:
    name = "internal_vesicle"
    requires_frames = True

    def enabled(self, config: EdgeQCConfig) -> bool:
        return config.internal_vesicle.enabled

    def run(self, detections, config, frames) -> QCCheckOutcome:
        if frames is None:
            raise ValueError("Internal-vesicle QC requires source video frames.")
        result = check_internal_vesicle_selection(frames, detections, config)
        flags = (
            frozenset({TrajectoryQCFlag.INTERNAL_VESICLE})
            if result.persistent_enclosing_boundary
            else frozenset()
        )
        return QCCheckOutcome(internal_vesicle=result, trajectory_flags=flags)


QC_CHECKS: tuple[QCCheck, ...] = (
    CurvatureCheck(),
    MinimumRadiusCheck(),
    LocalizedDeviationCheck(),
    AreaDeviationCheck(),
    InternalVesicleCheck(),
)


def run_configured_qc_checks(
    detections: list[EdgeDetection],
    config: EdgeQCConfig,
    frames: Frames = None,
) -> QCCheckOutcome:
    """Run the explicit ordered registry and combine aggregate outcomes."""
    outcome = QCCheckOutcome()
    for check in QC_CHECKS:
        if not check.enabled(config):
            continue
        if check.requires_frames and frames is None:
            raise ValueError(
                f"{check.name.replace('_', '-').capitalize()} QC is enabled "
                "but source video frames were not supplied."
            )
        check_outcome = check.run(detections, config, frames)
        outcome = QCCheckOutcome(
            curvature=check_outcome.curvature or outcome.curvature,
            area=check_outcome.area or outcome.area,
            internal_vesicle=(
                check_outcome.internal_vesicle or outcome.internal_vesicle
            ),
            trajectory_flags=(
                outcome.trajectory_flags | check_outcome.trajectory_flags
            ),
        )
    return outcome
