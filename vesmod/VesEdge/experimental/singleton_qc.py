"""Geometry-only QC for isolated radial contour excursions."""

from dataclasses import dataclass

import numpy as np

from ..qc.contract import QCCheck
from ..contour_geometry import fit_radial_baseline
from ..models import EdgeDetection
from vesmod.validation import require_integer_valued, require_nonnegative_real


CHECK_NAME = "singleton_deviation"


@dataclass(frozen=True)
class SingletonDeviationQCConfig:
    """Configuration owned by the singleton-deviation QC check."""

    enabled: bool = False
    order: int = 3
    min_residual_fraction: float = 0.05
    max_width_samples: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        order = require_integer_valued(self.order, "order")
        width = require_integer_valued(self.max_width_samples, "max_width_samples")
        if order < 0:
            raise ValueError("order must be non-negative.")
        if width <= 0:
            raise ValueError("max_width_samples must be positive.")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "max_width_samples", width)
        object.__setattr__(self, "min_residual_fraction", require_nonnegative_real(self.min_residual_fraction, "min_residual_fraction"))


@dataclass(frozen=True)
class SingletonDeviationQCResult:
    """Per-trajectory outcome produced by singleton-deviation QC."""

    scores: tuple[float, ...]
    counts: tuple[int, ...]
    rejected_count: int


def check_singleton_deviation(
    edge: EdgeDetection,
    order: int,
    min_residual_fraction: float,
    max_width_samples: int,
) -> None:
    """Reject contours containing narrow excursions from a smooth baseline."""
    radii = np.asarray(edge.analysis_contour.r, dtype=float)
    median = float(np.median(radii))
    baseline = fit_radial_baseline(radii, order).values
    residual = np.abs(radii - baseline) / median
    above = residual >= min_residual_fraction
    # Count contiguous threshold crossings on a circular contour.
    doubled = np.concatenate((above, above))
    runs = []
    start = None
    for index, value in enumerate(doubled):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(doubled) - start))
    widths = [width for start, width in runs if start < radii.size and start + width > radii.size]
    widths.extend(width for start, width in runs if start < radii.size and start + width <= radii.size)
    singleton_count = sum(width <= max_width_samples for width in widths)
    edge.qc.diagnostics["singleton_count"] = singleton_count
    edge.qc.diagnostics["singleton_score"] = float(np.max(residual)) if residual.size else 0.0
    if singleton_count:
        edge.qc.flags.add(CHECK_NAME)
    else:
        edge.qc.flags.discard(CHECK_NAME)


class SingletonDeviationCheck(QCCheck):
    """Apply narrow singleton-deviation QC to each detection."""
    name = CHECK_NAME
    config_type = SingletonDeviationQCConfig
    requires_source_video_frames = False

    def default_config(self):
        return SingletonDeviationQCConfig()

    def config_to_dict(self, config):
        return {"enabled": config.enabled, "order": config.order, "min_residual_fraction": config.min_residual_fraction, "max_width_samples": config.max_width_samples}

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
                CHECK_NAME in detection.qc.flags
                for detection in detections
            ),
        )

    def add_cli_arguments(self, parser) -> None:
        """Register command-line options for this check."""
        parser.add_argument("--singleton-deviation-qc", action="store_true", help="Enable narrow, isolated radial-excursion QC.")
        parser.add_argument("--singleton-deviation-order", type=int, default=3, help="Fourier baseline order for singleton-deviation QC. Default: 3.")
        parser.add_argument("--min-singleton-deviation-residual-fraction", type=float, default=0.05, help="Minimum residual divided by median radius for singleton-deviation QC. Default: 0.05.")
        parser.add_argument("--max-singleton-deviation-width-samples", type=int, default=2, help="Maximum contiguous excursion width in angular samples. Default: 2.")

    def config_from_args(self, args):
        """Build settings from parsed command-line options."""
        return SingletonDeviationQCConfig(
            enabled=getattr(args, "singleton_deviation_qc", False),
            order=getattr(args, "singleton_deviation_order", 3),
            min_residual_fraction=getattr(args, "min_singleton_deviation_residual_fraction", 0.05),
            max_width_samples=getattr(args, "max_singleton_deviation_width_samples", 2),
        )

    summary_order = 3
    summary_defaults = {"singleton_deviation_rejected": 0}

    def summary_values(self, edges):
        """Count frames rejected by this check."""
        return {
            "singleton_deviation_rejected": sum(
                CHECK_NAME in detection.qc.flags
                for detection in edges.successful_detections
            )
        }
