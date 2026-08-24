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

from .config import EdgeQCConfig


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
    POPULATION_OUTLIER = auto()


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
        Curvature score for each successfully extracted detection, in detection
        order. Non-finite contours are represented by ``nan``.
    rejected_count : int
        Number of detections rejected by curvature QC.
    """

    scores: tuple[float, ...]
    rejected_count: int


@dataclass(frozen=True)
class EdgePopulationResult:
    """Results from trajectory-level center/radius population analysis.

    Attributes
    ----------
    bic_one_population : float | None
        BIC from the one-component Gaussian mixture model. None if there were
        too few detections to perform population analysis.
    bic_two_populations : float | None
        BIC from the two-component Gaussian mixture model. None if there were
        too few detections to perform population analysis.
    two_populations_detected : bool
        Whether the two-component model was sufficiently favored over the
        one-component model.
    population_sizes : tuple[int, ...]
        Number of detections assigned to each detected population.
    rejected_population : int | None
        Label of the population rejected as a minor outlier population.
        None if no population was rejected.
    delta_bic : float | None
        Improvement in BIC obtained by fitting two populations rather than
        one. Positive values favor two populations.
    """

    bic_one_population: float | None
    bic_two_populations: float | None
    two_populations_detected: bool
    population_sizes: tuple[int, ...]
    rejected_population: int | None

    @property
    def delta_bic(self) -> float | None:
        """Return the BIC improvement from fitting two populations."""
        if (
            self.bic_one_population is None
            or self.bic_two_populations is None
        ):
            return None

        return self.bic_one_population - self.bic_two_populations


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
    population : EdgePopulationResult | None
        Summary of trajectory-level population QC. None when population QC was
        disabled.
    """

    config: EdgeQCConfig
    curvature: CurvatureQCResult | None
    population: EdgePopulationResult | None
