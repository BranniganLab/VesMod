"""Composable built-in VesEdge quality-control checks."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Protocol

from numpy.typing import NDArray

from .area_qc import AreaQCConfig, check_area_deviation
from .edge_filtering import CurvatureQCConfig, CurvatureQCResult, check_curvature
from .experimental.internal_vesicle_qc import (
    InternalVesicleQCConfig,
    InternalVesicleQCResult,
    check_internal_vesicle_selection,
)
from .frame_source import FrameSource
from .localized_deviation_qc import (
    LocalizedDeviationQCConfig,
    LocalizedDeviationQCResult,
    check_localized_deviation,
)
from .minimum_radius_qc import (
    MinimumRadiusQCConfig,
    MinimumRadiusQCResult,
    check_minimum_radius,
)
from .qc_config import EdgeQCConfig
from .singleton_qc import (
    SingletonDeviationQCConfig,
    SingletonDeviationQCResult,
    check_singleton_deviation,
)
from .models import EdgeDetection, QCFlag, TrajectoryQCFlag


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

    def trajectory_flags(self, result: object | None) -> frozenset[TrajectoryQCFlag]:
        """Return trajectory flags represented by this check's result."""


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

    def run(self, detections, config, frames) -> MinimumRadiusQCResult:
        """Apply the minimum-radius check to each detection."""
        del frames
        for detection in detections:
            check_minimum_radius(
                detection,
                min_median_radius_pixels=config.min_median_radius_pixels,
            )
        return MinimumRadiusQCResult(
            median_radii_pixels=tuple(
                float(detection.qc.diagnostics["median_radius_pixels"])
                for detection in detections
            ),
            rejected_count=sum(
                QCFlag.MINIMUM_RADIUS in detection.qc.flags
                for detection in detections
            ),
        )


class LocalizedDeviationCheck:
    """Apply localized-deviation quality control to each detection."""

    name = "localized_deviation"
    config_type = LocalizedDeviationQCConfig
    requires_frames = False
    def default_config(self): return LocalizedDeviationQCConfig()
    def config_to_dict(self, config):
        return {
            "enabled": config.enabled,
            "order": config.order,
            "max_residual_fraction": config.max_residual_fraction,
            "support_residual_fraction": config.support_residual_fraction,
            "max_support_samples": config.max_support_samples,
        }

    def enabled(self, config) -> bool:
        """Return whether localized-deviation quality control is enabled."""
        return config.enabled

    def run(self, detections, config, frames) -> LocalizedDeviationQCResult:
        """Apply the localized-deviation check to each detection."""
        del frames
        for detection in detections:
            check_localized_deviation(
                detection,
                config.order,
                config.max_residual_fraction,
                config.support_residual_fraction,
                config.max_support_samples,
            )
        return LocalizedDeviationQCResult(
            scores=tuple(
                float(detection.qc.diagnostics["localized_deviation_score"])
                for detection in detections
            ),
            support_samples=tuple(
                detection.qc.diagnostics["localized_deviation_support_samples"]
                for detection in detections
            ),
            rejected_count=sum(
                QCFlag.LOCALIZED_DEVIATION in detection.qc.flags
                for detection in detections
            ),
        )


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

    def run(self, detections, config, frames) -> SingletonDeviationQCResult:
        """Apply singleton-deviation QC to each detection."""
        del frames
        for detection in detections:
            check_singleton_deviation(
                detection,
                config.order, config.min_residual_fraction, config.max_width_samples,
            )
        return SingletonDeviationQCResult(
            scores=tuple(
                float(detection.qc.diagnostics["singleton_score"])
                for detection in detections
            ),
            counts=tuple(
                int(detection.qc.diagnostics["singleton_count"])
                for detection in detections
            ),
            rejected_count=sum(
                QCFlag.SINGLETON_DEVIATION in detection.qc.flags
                for detection in detections
            ),
        )


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

    def trajectory_flags(
        self, result: InternalVesicleQCResult | None,
    ) -> frozenset[TrajectoryQCFlag]:
        """Map persistent enclosing-boundary evidence to a trajectory flag."""
        if result is not None and result.persistent_enclosing_boundary:
            return frozenset({TrajectoryQCFlag.INTERNAL_VESICLE})
        return frozenset()


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
        flag_provider = getattr(check, "trajectory_flags", None)
        if flag_provider is not None:
            trajectory_flags.update(flag_provider(result))
    return QCCheckOutcome(results, frozenset(trajectory_flags))


def config_from_dict(values: dict) -> EdgeQCConfig:
    """Deserialize current nested data or the previous explicit schema."""
    specs = {check.name: check for check in QC_CHECKS}
    if "curvature_threshold" in values:
        legacy_fields = {
            "curvature_threshold", "enable_curvature_qc",
            "max_relative_area_deviation", "enable_area_qc",
            "enable_internal_vesicle_qc", "max_internal_vesicle_area_fraction",
            "internal_vesicle_min_radius_ratio",
            "internal_vesicle_min_separation_pixels",
            "internal_vesicle_min_separation_fraction",
            "internal_vesicle_gradient_ratio",
            "internal_vesicle_max_radial_deviation_fraction",
            "internal_vesicle_min_angular_coverage", "internal_vesicle_max_frames",
            "internal_vesicle_min_valid_frames",
            "internal_vesicle_min_valid_frame_fraction",
            "internal_vesicle_min_frame_fraction",
        }
        unknown = set(values) - legacy_fields
        if unknown:
            raise TypeError(f"Unexpected QC configuration field: {sorted(unknown)[0]}")
        if "internal_vesicle_min_separation_pixels" in values:
            raise ValueError(
                "Legacy internal_vesicle_min_separation_pixels cannot be converted "
                "without a contour radius; use internal_vesicle_min_separation_fraction."
            )
        values = {
            "curvature": CurvatureQCConfig(
                threshold=values["curvature_threshold"],
                enabled=values.get("enable_curvature_qc", True),
            ),
            "area": AreaQCConfig(
                max_relative_deviation=values.get("max_relative_area_deviation", 0.25),
                enabled=values.get("enable_area_qc", True),
            ),
            "minimum_radius": MinimumRadiusQCConfig(),
            "localized_deviation": LocalizedDeviationQCConfig(),
            "singleton_deviation": SingletonDeviationQCConfig(),
            "internal_vesicle": InternalVesicleQCConfig(
                enabled=values.get("enable_internal_vesicle_qc", False),
                max_area_fraction=values.get("max_internal_vesicle_area_fraction", 0.5),
                min_radius_ratio=values.get("internal_vesicle_min_radius_ratio", 1.15),
                min_separation_fraction=values.get("internal_vesicle_min_separation_fraction", 0.4),
                gradient_ratio=values.get("internal_vesicle_gradient_ratio", 0.5),
                max_radial_deviation_fraction=values.get("internal_vesicle_max_radial_deviation_fraction", 0.15),
                min_angular_coverage=values.get("internal_vesicle_min_angular_coverage", 0.6),
                max_frames=values.get("internal_vesicle_max_frames", 20),
                min_valid_frames=values.get("internal_vesicle_min_valid_frames", 3),
                min_valid_frame_fraction=values.get("internal_vesicle_min_valid_frame_fraction", 0.5),
                min_frame_fraction=values.get("internal_vesicle_min_frame_fraction", 0.5),
            ),
        }
        return EdgeQCConfig(checks=values)
    values = dict(values)
    aliases = {
        "radius": "minimum_radius",
        "baseline": "localized_deviation",
        "singleton": "singleton_deviation",
    }
    for alias, name in aliases.items():
        if alias not in values:
            continue
        if name in values:
            raise TypeError(
                f"QC configuration cannot contain both {alias} and {name}."
            )
        values[name] = values.pop(alias)
    unknown = set(values) - set(specs)
    if unknown:
        raise TypeError(f"Unexpected QC configuration field: {sorted(unknown)[0]}")
    parsed = {
        name: (
            spec.config_type(**values[name])
            if name in values
            else spec.default_config()
        )
        for name, spec in specs.items()
    }
    return EdgeQCConfig(checks=parsed)
