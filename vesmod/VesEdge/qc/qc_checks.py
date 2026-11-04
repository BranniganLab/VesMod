"""Ordered registry and execution of VesEdge quality-control checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from numpy.typing import NDArray

from ..io.frame_source import OnDemandFrameSequence
from ..models import EdgeDetection
from ..experimental.area_qc import AreaDeviationCheck
from .checks.edge_filtering import CurvatureCheck
from .checks.localized_deviation_qc import LocalizedDeviationCheck
from ..experimental.minimum_radius_qc import MinimumRadiusCheck
from ..experimental.singleton_qc import SingletonDeviationCheck
from .contract import QCCheck
from .qc_config import VesicleQCConfig


Frames = OnDemandFrameSequence | NDArray | None


@dataclass(frozen=True)
class QCCheckOutcome:
    """Results returned by registered checks and trajectory flags."""

    results: dict[str, object] = field(default_factory=dict)
    trajectory_flags: frozenset[str] = frozenset()


QC_CHECKS: tuple[QCCheck, ...] = (
    CurvatureCheck(),
    MinimumRadiusCheck(),
    LocalizedDeviationCheck(),
    SingletonDeviationCheck(),
    AreaDeviationCheck(),
)


def checks_requiring_frames(config: VesicleQCConfig) -> tuple[QCCheck, ...]:
    """Return enabled checks that need source images."""
    return tuple(
        check for check in QC_CHECKS
        if check.requires_frames and check.enabled(config.get_config(check.name))
    )


def run_configured_qc_checks(
    detections: list[EdgeDetection],
    config: VesicleQCConfig,
    frames: Frames = None,
) -> QCCheckOutcome:
    """Run the explicit ordered registry and combine aggregate outcomes."""
    results = {}
    trajectory_flags = set()
    for check in QC_CHECKS:
        check_config = config.get_config(check.name)
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
        trajectory_flags.update(check.trajectory_flags(result))
    return QCCheckOutcome(results, frozenset(trajectory_flags))


def config_from_dict(values: dict) -> VesicleQCConfig:
    """Deserialize check-keyed configuration."""
    specs = {check.name: check for check in QC_CHECKS}
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
    return VesicleQCConfig(checks=parsed)
