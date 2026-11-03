"""Interface implemented by each registered VesEdge quality check."""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Mapping
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
    requires_frames = False
    summary_order = 5
    summary_defaults: Mapping[str, object] = {}
    diagnostic_writers: tuple[tuple[str, Callable[[Path, VesicleEdges], None]], ...] = ()

    @abstractmethod
    def default_config(self) -> object:
        """Return validated settings for use when the caller omits this check."""

    @abstractmethod
    def config_to_dict(self, config: object) -> dict:
        """Serialize settings for replay."""

    @abstractmethod
    def enabled(self, config: object) -> bool:
        """Return whether the check should run."""

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

    def summary_values(self, edges: VesicleEdges) -> dict[str, object]:
        """Return measured values for the check's declared summary columns."""
        return {}
