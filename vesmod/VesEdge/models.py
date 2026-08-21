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


@dataclass(frozen=True)
class ImageContour:
    """
    A vesicle contour expressed in the coordinate system of an image.

    The radial coordinates are assumed to be sampled at evenly spaced angular
    positions spanning 0 to 2π, not including 2π itself.

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
        """
        Return the x-coordinates of the contour.

        The first coordinate is repeated at the end so the contour is closed
        when plotted.
        """
        x_vals = self.r * np.cos(self.theta) + self.origin[0]
        return np.append(x_vals, x_vals[0])

    @property
    def y(self) -> NDArray[np.float64]:
        """
        Return the y-coordinates of the contour.

        The first coordinate is repeated at the end so the contour is closed
        when plotted.
        """
        y_vals = self.r * np.sin(self.theta) + self.origin[1]
        return np.append(y_vals, y_vals[0])


class QCFlag(Enum):
    """Reasons a successfully extracted edge may fail quality control."""

    CURVATURE = auto()
    POPULATION_OUTLIER = auto()


FRAME_QC_FLAGS = {
    QCFlag.CURVATURE,
}

TRAJECTORY_QC_FLAGS = {
    QCFlag.POPULATION_OUTLIER,
}


@dataclass
class EdgeQC:
    """
    Quality-control information associated with one detected edge.

    Attributes
    ----------
    flags : set[QCFlag]
        QC checks that the edge has failed.
    curvature_score : float | None
        Maximum absolute wrapped finite second difference of the analysis
        contour. None if curvature QC has not been run.
    population_label : int | None
        Population assigned during trajectory-level population clustering.
        None if population QC has not been run.
    population_probability : float | None
        Posterior probability that the detection belongs to its assigned
        population. None if population QC has not been run.
    passed : bool
        Whether the edge has passed all QC checks that have been run.
    """

    flags: set[QCFlag] = field(default_factory=set)

    curvature_score: float | None = None

    population_label: int | None = None
    population_probability: float | None = None

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
    radii_microns : NDArray[np.float64]
        Radial profile from `analysis_contour` converted to physical units.
    qc : EdgeQC
        Quality-control information associated with this detection.
    median_radius : float
        Median physical radius of the analysis contour.
    accepted : bool
        Whether the detection has passed all QC checks run so far.
    """

    full_contour: ImageContour
    analysis_contour: ImageContour
    radii_microns: NDArray[np.float64]
    qc: EdgeQC = field(default_factory=EdgeQC)

    @property
    def median_radius(self) -> float:
        """Return the median radius of the analysis contour in microns."""
        return float(np.median(self.radii_microns))

    @property
    def accepted(self) -> bool:
        """Return whether this edge detection has passed quality control."""
        return self.qc.passed


@dataclass(frozen=True)
class EdgeDetectionFailure:
    """
    Record a frame for which edge extraction failed with an exception.

    Attributes
    ----------
    error : str
        Error message produced while attempting edge extraction.
    """

    error: str


EdgeResult = EdgeDetection | EdgeDetectionFailure
