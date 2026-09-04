#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared data models for VesEdge.

Contains data structures representing detected contours, edge-extraction
results, and quality-control results.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from vesmod.validation import (
    require_finite_array,
    require_finite_real,
    require_numeric_array,
)

from .config import EdgeQCConfig


@dataclass(frozen=True)
class ImageContour:
    """
    A vesicle contour expressed in the coordinate system of an image.

    The radial coordinates are assumed to be sampled at evenly spaced angular
    positions spanning 0 to 2π, not including 2π itself. Successful
    construction guarantees a finite two-coordinate origin and a non-empty,
    one-dimensional, finite, positive real-valued radial array. The radial
    array is copied so mutation of the caller's input cannot invalidate the
    constructed contour.

    Attributes
    ----------
    origin : tuple[float, float]
        Cartesian coordinates (x, y) of the contour origin in image pixels.
    r : NDArray[np.float64]
        Radial distances from `origin` to the contour in image pixels.
    theta : NDArray[np.float64]
        Evenly spaced angular coordinates corresponding to `r`, ranging from
        0 to 2π, not including 2π.
    x : NDArray[np.float64]
        Cartesian x-coordinates of the contour. The first coordinate is
        repeated at the end to close the contour for plotting.
    y : NDArray[np.float64]
        Cartesian y-coordinates of the contour. The first coordinate is
        repeated at the end to close the contour for plotting.
    """

    origin: tuple[float, float]
    r: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and normalize contour coordinates."""
        if not isinstance(self.origin, tuple) or len(self.origin) != 2:
            raise TypeError("origin must be a two-coordinate tuple.")
        origin = tuple(
            require_finite_real(value, "origin coordinate")
            for value in self.origin
        )

        require_numeric_array(self.r, "r")
        if self.r.ndim != 1:
            raise ValueError("r must be a one-dimensional array.")
        if self.r.size == 0:
            raise ValueError("r must contain at least one radial sample.")
        if np.issubdtype(self.r.dtype, np.complexfloating):
            raise TypeError("r must contain real-valued numbers.")
        require_finite_array(self.r, "r")

        radii = np.asarray(self.r, dtype=float).copy()
        if np.any(radii <= 0):
            raise ValueError("r must contain only positive values.")

        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "r", radii)

    @property
    def theta(self) -> NDArray[np.float64]:
        """Return evenly spaced angular coordinates corresponding to `r`."""
        return np.linspace(
            0.0,
            2.0 * np.pi,
            self.r.shape[0],
            endpoint=False,
        )

    @property
    def x(self) -> NDArray[np.float64]:
        """Return x-coordinates, closing the contour for plotting."""
        x_vals = self.r * np.cos(self.theta) + self.origin[0]
        return np.append(x_vals, x_vals[0])

    @property
    def y(self) -> NDArray[np.float64]:
        """Return y-coordinates, closing the contour for plotting."""
        y_vals = self.r * np.sin(self.theta) + self.origin[1]
        return np.append(y_vals, y_vals[0])


class QCFlag(Enum):
    """Reasons a successfully extracted edge may fail quality control."""

    CURVATURE = auto()
    AREA_DEVIATION = auto()


@dataclass
class EdgeQC:
    """
    Quality-control information associated with one detected edge.

    Attributes
    ----------
    flags : set[QCFlag]
        QC checks that the edge has failed.
    curvature_score : float | None
        Maximum absolute wrapped finite second difference of the
        median-radius-normalized analysis contour. This score is dimensionless.
        None if curvature QC has not been run.
    area_pixels2 : float | None
        Area enclosed by the native contour in squared pixels. None if area QC
        has not been run.
    relative_area_deviation : float | None
        Absolute fractional deviation from the trajectory median contour area.
        None if area QC has not been run.
    passed : bool
        Whether the edge has passed all QC checks that have been run.
    """

    flags: set[QCFlag] = field(default_factory=set)
    curvature_score: float | None = None
    area_pixels2: float | None = None
    relative_area_deviation: float | None = None

    @property
    def passed(self) -> bool:
        """Return whether the edge has passed all QC checks run so far."""
        return not self.flags


@dataclass
class EdgeDetection:
    """
    Successfully detected vesicle edge from one video frame.

    Attributes
    ----------
    full_contour : ImageContour
        Detected contour at the native angular sampling of the edge extractor.
    analysis_contour : ImageContour
        Contour used for subsequent analysis. If downsampling was requested,
        this is the downsampled contour. Otherwise, this is the same contour
        as `full_contour`.
    qc : EdgeQC
        Quality-control information associated with this detection.
    frame_index : int | None
        Zero-based source video frame index. None only for legacy/manually
        constructed results that have not yet been associated with a
        :class:`VesicleEdges` trajectory.
    """

    full_contour: ImageContour
    analysis_contour: ImageContour
    qc: EdgeQC = field(default_factory=EdgeQC)
    frame_index: int | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class EdgeDetectionFailure:
    """
    Record a frame for which edge extraction failed with an exception.

    Attributes
    ----------
    error : str
        Error message produced while attempting edge extraction.
    frame_index : int | None
        Zero-based source video frame index. None only for legacy/manually
        constructed results that have not yet been associated with a
        :class:`VesicleEdges` trajectory.
    """

    error: str
    frame_index: int | None = field(default=None, kw_only=True)


EdgeResult = EdgeDetection | EdgeDetectionFailure


@dataclass(frozen=True)
class CurvatureQCResult:
    """Trajectory-level summary of frame curvature QC.

    Attributes
    ----------
    scores : tuple[float, ...]
        Dimensionless normalized-curvature score for each successfully
        extracted detection, in detection order. Non-finite contours are
        represented by ``nan``.
    rejected_count : int
        Number of detections rejected by curvature QC.
    """

    scores: tuple[float, ...]
    rejected_count: int


@dataclass(frozen=True)
class AreaQCResult:
    """Trajectory-level summary of contour-area deviation QC.

    Attributes
    ----------
    areas_pixels2 : tuple[float, ...]
        Enclosed area for each successful detection, in detection order.
    reference_area_pixels2 : float
        Median finite positive area among curvature-passing contours. This is
        nan when no contour passes curvature QC.
    relative_deviations : tuple[float, ...]
        Absolute fractional area deviation for each successful detection.
    rejected_count : int
        Number of detections rejected by area QC.
    """

    areas_pixels2: tuple[float, ...]
    reference_area_pixels2: float
    relative_deviations: tuple[float, ...]
    rejected_count: int


@dataclass(frozen=True)
class VesicleQCResult:
    """Aggregate results from one completed VesEdge QC run.

    Attributes
    ----------
    config : EdgeQCConfig
        Configuration used for the QC run.
    curvature : CurvatureQCResult | None
        Summary of frame-level curvature QC. None when curvature QC was
        disabled.
    area : AreaQCResult | None
        Summary of trajectory-level contour-area QC. None when area QC was
        disabled.
    """

    config: EdgeQCConfig
    curvature: CurvatureQCResult | None
    area: AreaQCResult | None = None
