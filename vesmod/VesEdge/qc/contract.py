"""Interface implemented by each registered VesEdge quality check."""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import EdgeDetection
    from ..vesicle_edges import VesicleEdges
    from .qc_checks import Frames


class QCCheck(ABC):
    """Own check settings, evaluation, and any command-line reporting."""

    name: str
    config_type: type
    requires_source_video_frames = False
    summary_order = 5
    diagnostic_writers: tuple[tuple[str, Callable[[Path, VesicleEdges], None]], ...] = ()

    def default_config(self) -> object:
        """Construct settings using the configuration class's defaults."""
        return self.config_type()

    def config_to_dict(self, config) -> dict:
        """Serialize dataclass settings for replay."""
        return asdict(config)

    def enabled(self, config) -> bool:
        """Return whether the check's configuration enables it."""
        return config.enabled

    @abstractmethod
    def run(
        self, detections: list[EdgeDetection], config: object, frames: Frames,
    ) -> object | None:
        """Evaluate detections and optionally return check-owned diagnostics."""

    def trajectory_flags(self, result: object | None) -> frozenset[str]:
        """Return whole-recording rejection reasons represented by the result."""
        return frozenset()

    def add_cli_arguments(self, parser: ArgumentParser) -> None:
        """Register any command-line options for this check."""

    def config_from_args(self, args: Namespace) -> object:
        """Construct validated settings from command-line arguments."""
        return self.default_config()

    @property
    def summary_defaults(self) -> Mapping[str, object]:
        """Declare the default column for this check's rejected-frame count."""
        return {f"{self.name}_rejected": 0}

    def summary_values(self, edges: VesicleEdges) -> dict[str, object]:
        """Count frames whose rejection flags contain this check's name."""
        return {
            f"{self.name}_rejected": sum(
                self.name in detection.qc.flags
                for detection in edges.successful_detections
            )
        }
