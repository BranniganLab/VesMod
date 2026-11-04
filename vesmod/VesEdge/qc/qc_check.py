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
    from .qc_runner import Frames


class QCCheck(ABC):
    """Own check settings, evaluation, and any command-line reporting.

    Subclasses must define `name`, `config_type`, and `run`. The first two
    can be class attributes. Override optional hooks when the check needs
    command-line options or different reporting behavior.
    """

    requires_source_video_frames = False
    summary_order = 5
    diagnostic_writers: tuple[tuple[str, Callable[[Path, VesicleEdges], None]], ...] = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Define the unique registry name as a class attribute or property."""
        raise NotImplementedError

    @property
    @abstractmethod
    def config_type(self) -> type:
        """Define the settings class as a class attribute or property."""
        raise NotImplementedError

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
        """Evaluate detections, record rejections, and return diagnostics.

        `detections` contains successful contour extractions, `config` is
        this check's settings object, and `frames` provides source images
        when supplied. Record frame rejections in `detection.qc.flags` and
        measurements in `detection.qc.diagnostics`. Return the check's
        aggregate result, or `None` when there is no aggregate result.
        """
        raise NotImplementedError

    def trajectory_flags(self, result: object | None) -> frozenset[str]:
        """Override to return whole-recording rejection reasons for the result."""
        return frozenset()

    def add_cli_arguments(self, parser: ArgumentParser) -> None:
        """Override to add command-line options for this check."""
        pass

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
