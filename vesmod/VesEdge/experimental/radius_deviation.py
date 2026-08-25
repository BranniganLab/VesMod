"""Experimental median-radius screen for obvious wrong-object detections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from typing import Sequence

import numpy as np

from ..models import EdgeDetection


@dataclass(frozen=True)
class RadiusDeviationConfig:
    """Configure the experimental trajectory-level radius screen.

    Parameters
    ----------
    max_relative_deviation : float
        Largest accepted fractional difference between a detection's median
        radius and the trajectory-wide median radius. For example, ``0.2``
        permits a 20% difference.
    """

    max_relative_deviation: float

    def __post_init__(self) -> None:
        """Validate and normalize the experimental threshold."""
        value = self.max_relative_deviation
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("max_relative_deviation must be a real number.")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("max_relative_deviation must be finite.")
        if value < 0:
            raise ValueError("max_relative_deviation must be non-negative.")
        object.__setattr__(self, "max_relative_deviation", value)

    def to_dict(self) -> dict[str, float]:
        """Return JSON-serializable configuration values."""
        return {"max_relative_deviation": float(self.max_relative_deviation)}


@dataclass(frozen=True)
class RadiusDeviationFrame:
    """Experimental radius-screen result for one curvature-accepted frame."""

    frame_index: int
    median_radius_pixels: float
    relative_deviation: float
    accepted: bool

    def to_dict(self) -> dict:
        """Return JSON-serializable frame diagnostics."""
        return asdict(self)


@dataclass(frozen=True)
class RadiusDeviationResult:
    """Aggregate result of the experimental median-radius screen."""

    config: RadiusDeviationConfig
    reference_radius_pixels: float
    frames: tuple[RadiusDeviationFrame, ...]

    @property
    def accepted_positions(self) -> tuple[int, ...]:
        """Return positions accepted within the screened detection sequence."""
        return tuple(
            position
            for position, frame in enumerate(self.frames)
            if frame.accepted
        )

    @property
    def rejected_count(self) -> int:
        """Return the number of detections rejected by the screen."""
        return sum(not frame.accepted for frame in self.frames)

    @property
    def accepted_count(self) -> int:
        """Return the number of detections accepted by the screen."""
        return len(self.frames) - self.rejected_count

    def to_dict(self) -> dict:
        """Return complete JSON-serializable diagnostics."""
        return {
            "method": "median_radius_deviation",
            "config": self.config.to_dict(),
            "reference_radius_pixels": float(self.reference_radius_pixels),
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "frames": [frame.to_dict() for frame in self.frames],
        }


def screen_radius_deviations(
    detections: Sequence[EdgeDetection],
    config: RadiusDeviationConfig,
) -> RadiusDeviationResult:
    """Screen detections against their trajectory-wide median radius.

    The caller controls which detections are eligible. The VesEdge CLI passes
    only detections that have already passed stable curvature QC, keeping this
    experimental operation separate from the core QC state.

    Raises
    ------
    ValueError
        If no detections are supplied or any median radius is non-positive or
        non-finite.
    """
    if not detections:
        raise ValueError("Radius-deviation screening requires detections.")

    radii = np.asarray(
        [np.median(detection.analysis_contour.r) for detection in detections],
        dtype=float,
    )
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError(
            "Radius-deviation screening requires positive, finite radii."
        )

    reference = float(np.median(radii))
    deviations = np.abs(radii - reference) / reference
    frames = tuple(
        RadiusDeviationFrame(
            frame_index=(
                position
                if detection.frame_index is None
                else int(detection.frame_index)
            ),
            median_radius_pixels=float(radius),
            relative_deviation=float(deviation),
            accepted=bool(deviation <= config.max_relative_deviation),
        )
        for position, (detection, radius, deviation) in enumerate(zip(
            detections,
            radii,
            deviations,
            strict=True,
        ))
    )
    return RadiusDeviationResult(
        config=config,
        reference_radius_pixels=reference,
        frames=frames,
    )
