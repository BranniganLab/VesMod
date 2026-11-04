"""Geometry-only QC for isolated radial contour excursions."""

from dataclasses import dataclass

import numpy as np

from ..qc_check import QCCheck
from ..qc_result import QCResult
from ...contour_geometry import fit_radial_baseline
from ...models import EdgeDetection
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


def _singleton_measurements(edge, order, min_residual_fraction, max_width_samples):
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
    score = float(np.max(residual)) if residual.size else 0.0
    return singleton_count, score


def check_singleton_deviation(
    detections: list[EdgeDetection], order: int,
    min_residual_fraction: float, max_width_samples: int,
) -> QCResult:
    """Return narrow-excursion decisions, fractional scores, and counts."""
    measured = [
        _singleton_measurements(edge, order, min_residual_fraction, max_width_samples)
        for edge in detections
    ]
    return QCResult(
        check_name=CHECK_NAME,
        frame_indices=tuple(edge.frame_index for edge in detections),
        rejected=tuple(count > 0 for count, _ in measured),
        diagnostics={
            "singleton_score": tuple(score for _, score in measured),
            "singleton_count": tuple(count for count, _ in measured),
        },
    )


class SingletonDeviationCheck(QCCheck):
    """Apply narrow singleton-deviation QC to each detection."""
    name = CHECK_NAME
    diagnostic_fields = ('singleton_score', 'singleton_count')
    config_type = SingletonDeviationQCConfig

    def run(self, detections, config, frames) -> QCResult:
        """Apply singleton-deviation QC to each detection."""
        del frames
        return check_singleton_deviation(
            detections, config.order, config.min_residual_fraction, config.max_width_samples,
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
