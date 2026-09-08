"""Geometry-only QC against a low-order radial baseline."""

from dataclasses import dataclass

import numpy as np
from .models import EdgeDetection, QCFlag
from .contour_geometry import fit_radial_baseline
from vesmod.validation import require_integer_valued, require_nonnegative_real


@dataclass(frozen=True)
class LocalizedDeviationQCConfig:
    """Configuration owned by the localized-deviation QC check."""

    enabled: bool = False
    order: int = 3
    max_residual_fraction: float = 0.05
    support_residual_fraction: float | None = None
    max_support_samples: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        order = require_integer_valued(self.order, "order")
        if order < 0:
            raise ValueError("order must be non-negative.")
        object.__setattr__(self, "order", order)
        object.__setattr__(
            self,
            "max_residual_fraction",
            require_nonnegative_real(
                self.max_residual_fraction,
                "max_residual_fraction",
            ),
        )
        if (self.support_residual_fraction is None) != (
            self.max_support_samples is None
        ):
            raise ValueError(
                "support residual and maximum support must be configured together."
            )
        if self.support_residual_fraction is not None:
            object.__setattr__(
                self,
                "support_residual_fraction",
                require_nonnegative_real(
                    self.support_residual_fraction,
                    "support_residual_fraction",
                ),
            )
        if self.max_support_samples is not None:
            support = require_integer_valued(
                self.max_support_samples,
                "max_support_samples",
            )
            if support <= 0:
                raise ValueError("max_support_samples must be positive.")
            object.__setattr__(self, "max_support_samples", support)


@dataclass(frozen=True)
class LocalizedDeviationQCResult:
    """Per-trajectory outcome produced by localized-deviation QC."""

    scores: tuple[float, ...]
    support_samples: tuple[int | None, ...]
    rejected_count: int


def check_localized_deviation(
    edge: EdgeDetection,
    order: int,
    max_residual_fraction: float,
    support_residual_fraction: float | None = None,
    max_support_samples: int | None = None,
) -> None:
    """Reject contours whose largest baseline residual is excessive."""
    if order < 0 or not np.isfinite(max_residual_fraction) or max_residual_fraction < 0:
        raise ValueError("localized-deviation QC configuration must use finite non-negative values")
    if (support_residual_fraction is None) != (max_support_samples is None):
        raise ValueError("support residual and maximum support must be configured together")
    radii = np.asarray(edge.analysis_contour.r, dtype=float)
    median = float(np.median(radii))
    fitted = fit_radial_baseline(radii, order).values
    score = float(np.max(np.abs(radii - fitted)) / median)
    edge.qc.diagnostics["localized_deviation_score"] = score
    rejected = score >= max_residual_fraction
    support = None
    if support_residual_fraction is not None:
        residual = np.abs(radii - fitted) / median
        mask = np.concatenate((
            residual >= support_residual_fraction,
            residual >= support_residual_fraction,
        ))
        if np.all(mask):
            support = radii.size
        else:
            lengths = []
            start = None
            for index, value in enumerate(mask):
                if value and start is None:
                    start = index
                elif not value and start is not None:
                    lengths.append(index - start)
                    start = None
            support = min(max(lengths, default=0), radii.size)
        rejected = rejected or (
            score >= support_residual_fraction and support <= max_support_samples
        )
    edge.qc.diagnostics["localized_deviation_support_samples"] = support
    if rejected:
        edge.qc.flags.add(QCFlag.LOCALIZED_DEVIATION)
    else:
        edge.qc.flags.discard(QCFlag.LOCALIZED_DEVIATION)
