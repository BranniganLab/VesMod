"""Composable built-in VesEdge quality-control checks."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Protocol

from numpy.typing import NDArray

from .area_qc import check_area_deviation
from .config import (
    AreaQCConfig,
    CurvatureQCConfig,
    EdgeQCConfig,
    InternalVesicleQCConfig,
    LegacyEdgeQCConfig,
    LocalizedDeviationQCConfig,
    MinimumRadiusQCConfig,
    SingletonDeviationQCConfig,
)
from .edge_filtering import check_curvature
from .experimental.internal_vesicle_qc import check_internal_vesicle_selection
from .frame_source import FrameSource
from .localized_deviation_qc import check_localized_deviation
from .minimum_radius_qc import check_minimum_radius
from .singleton_qc import check_singleton_deviation
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
    """Results returned by registered checks and trajectory flags."""

    results: dict[str, object] = dataclass_field(default_factory=dict)
    trajectory_flags: frozenset[TrajectoryQCFlag] = frozenset()


class QCCheck(Protocol):
    """Typed contract for a configured built-in quality-control check."""

    name: str
    config_type: type
    requires_frames: bool

    def default_config(self) -> object:
        """Return the disabled or default validated configuration."""

    def config_to_dict(self, config: object) -> dict:
        """Serialize this check's validated configuration."""

    def enabled(self, config: object) -> bool:
        """Return whether this check is enabled by configuration."""

    def run(
        self,
        detections: list[EdgeDetection],
        config: object,
        frames: Frames,
    ) -> object | None:
        """Apply the check and return its typed result, when any."""


class CurvatureCheck:
    """Apply curvature-score quality control to individual detections."""

    name = "curvature"
    config_type = CurvatureQCConfig
    requires_frames = False

    def default_config(self):
        return CurvatureQCConfig(threshold=0.1, enabled=False)

    def config_to_dict(self, config):
        return {"threshold": config.threshold, "enabled": config.enabled}

    def enabled(self, config) -> bool:
        """Return whether curvature quality control is enabled."""
        return config.enabled

    def run(self, detections, config, frames):
        """Record curvature scores and flags for each detection."""
        del frames
        for detection in detections:
            check_curvature(detection, threshold=config.threshold)
        scores = tuple(
            float(detection.qc.diagnostics["curvature_score"])
            if "curvature_score" in detection.qc.diagnostics
            else float("nan")
            for detection in detections
        )
        return CurvatureQCResult(
            scores=scores,
            rejected_count=sum(QCFlag.CURVATURE in detection.qc.flags for detection in detections),
        )


class MinimumRadiusCheck:
    """Apply the configured minimum-radius check to each detection."""

    name = "minimum_radius"
    config_type = MinimumRadiusQCConfig
    requires_frames = False

    def default_config(self): return MinimumRadiusQCConfig()
    def config_to_dict(self, config): return {"enabled": config.enabled, "min_median_radius_pixels": config.min_median_radius_pixels}

    def enabled(self, config) -> bool:
        """Return whether minimum-radius quality control is enabled."""
        return config.enabled

    def run(self, detections, config, frames) -> QCCheckOutcome:
        """Apply the minimum-radius check to each detection."""
        del frames
        for detection in detections:
            check_minimum_radius(
                detection,
                min_median_radius_pixels=config.min_median_radius_pixels,
            )
        return None


class LocalizedDeviationCheck:
    """Apply localized-deviation quality control to each detection."""

    name = "localized_deviation"
    config_type = LocalizedDeviationQCConfig
    requires_frames = False
    def default_config(self): return LocalizedDeviationQCConfig()
    def config_to_dict(self, config): return {"enabled": config.enabled, "order": config.order, "max_residual_fraction": config.max_residual_fraction}

    def enabled(self, config) -> bool:
        """Return whether localized-deviation quality control is enabled."""
        return config.enabled

    def run(self, detections, config, frames) -> QCCheckOutcome:
        """Apply the localized-deviation check to each detection."""
        del frames
        for detection in detections:
            check_localized_deviation(
                detection,
                config.order, config.max_residual_fraction,
            )
        return None


class SingletonDeviationCheck:
    """Apply narrow singleton-deviation QC to each detection."""
    name = "singleton_deviation"
    config_type = SingletonDeviationQCConfig
    requires_frames = False
    def default_config(self): return SingletonDeviationQCConfig()
    def config_to_dict(self, config): return {"enabled": config.enabled, "order": config.order, "min_residual_fraction": config.min_residual_fraction, "max_width_samples": config.max_width_samples}

    def enabled(self, config) -> bool:
        """Return whether singleton-deviation QC is enabled."""
        return config.enabled

    def run(self, detections, config, frames) -> QCCheckOutcome:
        """Apply singleton-deviation QC to each detection."""
        del frames
        for detection in detections:
            check_singleton_deviation(
                detection,
                config.order, config.min_residual_fraction, config.max_width_samples,
            )
        return None


class AreaDeviationCheck:
    """Apply trajectory-wide area-deviation quality control."""

    name = "area"
    config_type = AreaQCConfig
    requires_frames = False
    def default_config(self): return AreaQCConfig()
    def config_to_dict(self, config): return {"max_relative_deviation": config.max_relative_deviation, "enabled": config.enabled}

    def enabled(self, config) -> bool:
        """Return whether area-deviation quality control is enabled."""
        return config.enabled

    def run(self, detections, config, frames):
        """Evaluate area deviation across the detection trajectory."""
        del frames
        return check_area_deviation(detections, config.max_relative_deviation)


class InternalVesicleCheck:
    """Apply frame-dependent internal-vesicle quality control."""

    name = "internal_vesicle"
    config_type = InternalVesicleQCConfig
    requires_frames = True
    def default_config(self): return InternalVesicleQCConfig()
    def config_to_dict(self, config): return dict(config.__dict__)

    def enabled(self, config) -> bool:
        """Return whether internal-vesicle quality control is enabled."""
        return config.enabled

    def run(self, detections, config, frames):
        """Evaluate internal-vesicle selection using source video frames."""
        if frames is None:
            raise ValueError("Internal-vesicle QC requires source video frames.")
        return check_internal_vesicle_selection(frames, detections, config)


QC_CHECKS: tuple[QCCheck, ...] = (
    CurvatureCheck(),
    MinimumRadiusCheck(),
    LocalizedDeviationCheck(),
    SingletonDeviationCheck(),
    AreaDeviationCheck(),
    InternalVesicleCheck(),
)


def run_configured_qc_checks(
    detections: list[EdgeDetection],
    config: EdgeQCConfig,
    frames: Frames = None,
) -> QCCheckOutcome:
    """Run the explicit ordered registry and combine aggregate outcomes."""
    results = {}
    trajectory_flags = set()
    for check in QC_CHECKS:
        check_config = (
            config.for_check(check.name)
            if hasattr(config, "for_check")
            else config
        )
        if not check.enabled(check_config):
            continue
        if check.requires_frames and frames is None:
            raise ValueError(
                f"{check.name.replace('_', '-').capitalize()} QC is enabled "
                "but source video frames were not supplied."
            )
        result = check.run(detections, check_config, frames)
        if result is not None:
            results[check.name] = result
        if (
            check.name == "internal_vesicle"
            and result is not None
            and result.persistent_enclosing_boundary
        ):
            trajectory_flags.add(TrajectoryQCFlag.INTERNAL_VESICLE)
    return QCCheckOutcome(results, frozenset(trajectory_flags))


def config_from_dict(values: dict) -> EdgeQCConfig:
    """Deserialize current nested data or the previous explicit schema."""
    specs = {check.name: check for check in QC_CHECKS}
    if "curvature_threshold" in values:
        legacy = LegacyEdgeQCConfig._from_legacy_dict(values)
        values = {name: getattr(legacy, name) for name in specs}
        return EdgeQCConfig(checks=values)
    unknown = set(values) - set(specs)
    if unknown:
        raise TypeError(f"Unexpected QC configuration field: {sorted(unknown)[0]}")
    parsed = {
        name: spec.config_type(**values.get(name, {}))
        for name, spec in specs.items()
    }
    return EdgeQCConfig(checks=parsed)
