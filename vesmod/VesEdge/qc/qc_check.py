"""Interface implemented by each registered VesEdge quality check."""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .qc_result import QCResult

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

    diagnostic_fields: tuple[str, ...] = ()
    extra_result_fields: tuple[str, ...] = ()
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
    ) -> QCResult:
        """Return standard decisions and declared measurements for every frame.

        The runner validates the result and applies its decisions and
        diagnostics to the detections. Source images are supplied in frames
        when requested by requires_source_video_frames.
        """
        raise NotImplementedError

    def make_result(self, detections, *, extras=None) -> QCResult:
        """Collect this check's decisions and declared measurements."""
        return QCResult.from_detections(
            self.name, detections, self.diagnostic_fields, extras=extras,
        )

    def validate_result(self, result: QCResult) -> None:
        """Require standard decisions and explicitly declared measurements."""
        if type(result) is not QCResult:
            raise TypeError(f"{self.name} must return a QCResult.")
        if result.check_name != self.name:
            raise ValueError(f"QC result name must be {self.name!r}.")
        if set(result.diagnostics) != set(self.diagnostic_fields):
            raise ValueError(f"{self.name} result diagnostics do not match diagnostic_fields.")
        if set(result.extras) != set(self.extra_result_fields):
            raise ValueError(f"{self.name} result extras do not match extra_result_fields.")

    def result_from_dict(self, values: dict) -> QCResult:
        """Load standard result data and check its declared measurements."""
        result = QCResult.from_dict(values)
        self.validate_result(result)
        return result

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
