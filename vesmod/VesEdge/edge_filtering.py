#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality-control routines for VesEdge edge detections.

This module contains QC configuration and result objects, frame-level QC
functions, and trajectory-level QC functions. It does not perform edge
extraction.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, TypeGuard

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates, sobel
from sklearn.mixture import GaussianMixture

from .vesicle_video_utils import measure_wrapped_finite_second_difference


class QCFlag(Enum):
    """Reasons an extracted vesicle edge may be rejected."""

    CURVATURE = auto()
    IMAGE_SUPPORT = auto()
    POPULATION_OUTLIER = auto()


@dataclass
class EdgeQC:
    """
    Quality-control information associated with one detected edge.

    Attributes
    ----------
    flags : set[QCFlag]
        QC checks that the edge failed.
    curvature_score : float or None
        Maximum absolute wrapped finite second difference of the analysis
        contour. None if curvature QC has not been run.
    image_support_fraction : float or None
        Fraction of angular positions at which the detected edge has sufficient
        image-gradient support. None if image-support QC has not been run.
    minimum_image_support : float or None
        Minimum relative image-gradient support across the detected contour.
        None if image-support QC has not been run.
    population_label : int or None
        Population assigned to the detection by trajectory-level clustering.
        None if population QC has not been run.
    population_probability : float or None
        Posterior probability of membership in the assigned population.
        None if population QC has not been run.
    passed : bool
        Whether the edge has passed all QC checks that have been run.
    """

    flags: set[QCFlag] = field(default_factory=set)
    curvature_score: float | None = None
    image_support_fraction: float | None = None
    minimum_image_support: float | None = None
    population_label: int | None = None
    population_probability: float | None = None

    @property
    def passed(self) -> bool:
        """Whether the edge has passed all QC checks that have been run."""
        return not self.flags


@dataclass(frozen=True)
class EdgeQCConfig:
    """
    Configuration parameters for VesEdge quality-control checks.

    Parameters
    ----------
    curvature_threshold : float
        Maximum allowed absolute wrapped finite second difference in the
        analysis contour.
    image_support_threshold : float
        Minimum relative radial-gradient strength required at every angular
        position along the detected edge. Must be between 0 and 1.
    population_bic_threshold : float
        Minimum reduction in Bayesian information criterion (BIC) required
        for a two-population Gaussian mixture model to be preferred over a
        one-population model.
    max_minor_population_fraction : float
        Largest fraction of otherwise accepted detections that may belong to
        the smaller population for that population to be automatically
        rejected.
    image_support_search_radius : int
        Number of pixels inward and outward from the detected edge to search
        for the strongest local radial image gradient.
    """

    curvature_threshold: float = 5.0
    image_support_threshold: float = 0.5
    population_bic_threshold: float = 10.0
    max_minor_population_fraction: float = 0.25
    image_support_search_radius: int = 5

    def __post_init__(self) -> None:
        """Validate QC configuration parameters."""
        if not np.isfinite(self.curvature_threshold):
            raise ValueError("curvature_threshold must be finite.")
        if self.curvature_threshold < 0:
            raise ValueError("curvature_threshold must be non-negative.")

        if not np.isfinite(self.image_support_threshold):
            raise ValueError("image_support_threshold must be finite.")
        if not 0 <= self.image_support_threshold <= 1:
            raise ValueError(
                "image_support_threshold must be between 0 and 1."
            )

        if not np.isfinite(self.population_bic_threshold):
            raise ValueError("population_bic_threshold must be finite.")
        if self.population_bic_threshold < 0:
            raise ValueError(
                "population_bic_threshold must be non-negative."
            )

        if not np.isfinite(self.max_minor_population_fraction):
            raise ValueError(
                "max_minor_population_fraction must be finite."
            )
        if not 0 <= self.max_minor_population_fraction < 0.5:
            raise ValueError(
                "max_minor_population_fraction must be greater than or "
                "equal to 0 and less than 0.5."
            )

        if not isinstance(self.image_support_search_radius, int):
            raise TypeError(
                "image_support_search_radius must be an int."
            )
        if self.image_support_search_radius < 1:
            raise ValueError(
                "image_support_search_radius must be positive."
            )


@dataclass(frozen=True)
class EdgePopulationResult:
    """
    Results from center/radius population analysis.

    Attributes
    ----------
    bic_one_population : float or None
        BIC for the one-component Gaussian mixture model. None if there were
        too few detections to fit the population models.
    bic_two_populations : float or None
        BIC for the two-component Gaussian mixture model. None if there were
        too few detections to fit the population models.
    two_populations_detected : bool
        Whether the two-component model was sufficiently favored according to
        the configured BIC threshold.
    population_sizes : tuple[int, ...]
        Number of detections assigned to each detected population.
    rejected_population : int or None
        Population rejected as a minor outlier population. None if no
        population was automatically rejected.
    """

    bic_one_population: float | None
    bic_two_populations: float | None
    two_populations_detected: bool
    population_sizes: tuple[int, ...]
    rejected_population: int | None

    @property
    def delta_bic(self) -> float | None:
        """
        Improvement in BIC obtained by fitting two populations.

        Positive values favor the two-population model.
        """
        if (
            self.bic_one_population is None
            or self.bic_two_populations is None
        ):
            return None

        return self.bic_one_population - self.bic_two_populations


class _ImageContourLike(Protocol):
    """Structural type required from an image-space contour."""

    origin: tuple[float, float]
    r: NDArray[np.float64]

    @property
    def theta(self) -> NDArray[np.float64]:
        """Angular coordinates of the contour."""
        ...


class _EdgeDetectionLike(Protocol):
    """Structural type required from an edge detection."""

    full_contour: _ImageContourLike
    analysis_contour: _ImageContourLike
    radii_microns: NDArray[np.float64]
    qc: EdgeQC

    @property
    def median_radius(self) -> float:
        """Median radius of the analysis contour in physical units."""
        ...


def _is_edge_detection(value: object) -> TypeGuard[_EdgeDetectionLike]:
    """
    Determine whether an extraction result contains a detected edge.

    This structural check allows this module to distinguish successful edge
    detections from EdgeDetectionFailure objects without importing either
    class from vesicle_video and creating a circular import.
    """
    return (
        hasattr(value, "full_contour")
        and hasattr(value, "analysis_contour")
        and hasattr(value, "radii_microns")
        and hasattr(value, "qc")
        and hasattr(value, "median_radius")
    )


def check_curvature(
    edge: _EdgeDetectionLike,
    threshold: float,
) -> None:
    """
    Check an edge for excessive local curvature.

    The curvature metric is the largest absolute wrapped finite second
    difference of the radial values in the analysis contour. The edge is
    flagged if this value is greater than or equal to `threshold`.

    Parameters
    ----------
    edge : EdgeDetection
        Edge detection to evaluate.
    threshold : float
        Maximum allowed absolute wrapped finite second difference.

    Returns
    -------
    None
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    finite_second_difference = measure_wrapped_finite_second_difference(
        edge.analysis_contour.r
    )

    if not np.all(np.isfinite(finite_second_difference)):
        edge.qc.curvature_score = np.nan
        edge.qc.flags.add(QCFlag.CURVATURE)
        return

    curvature_score = float(
        np.max(np.abs(finite_second_difference))
    )
    edge.qc.curvature_score = curvature_score

    if curvature_score >= threshold:
        edge.qc.flags.add(QCFlag.CURVATURE)
    else:
        edge.qc.flags.discard(QCFlag.CURVATURE)


def check_image_support(
    frame: NDArray[np.float64],
    edge: _EdgeDetectionLike,
    threshold: float,
    search_radius: int = 5,
) -> None:
    """
    Check whether the detected contour is supported by image gradients.

    For each angular position along the full detected contour, the image
    gradient is projected onto the radial direction. The radial-gradient
    magnitude at the detected edge is divided by the largest radial-gradient
    magnitude within `search_radius` pixels of that edge.

    A relative support value of 1 indicates that the detected edge lies on
    the strongest radial gradient in the local search window. The edge fails
    this QC check if any angular position has relative support below
    `threshold`.

    Parameters
    ----------
    frame : NDArray[np.float64]
        Two-dimensional image from which the edge was extracted.
    edge : EdgeDetection
        Edge detection to evaluate.
    threshold : float
        Minimum relative radial-gradient strength required at every angular
        position. Must be between 0 and 1.
    search_radius : int, optional
        Number of pixels inward and outward from the detected edge over which
        to search for a stronger radial gradient. Default is 5.

    Returns
    -------
    None
    """
    if frame.ndim != 2:
        raise ValueError("frame must be a 2D array.")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    if not isinstance(search_radius, int):
        raise TypeError("search_radius must be an int.")

    if search_radius < 1:
        raise ValueError("search_radius must be positive.")

    contour = edge.full_contour

    if not np.all(np.isfinite(contour.r)):
        edge.qc.image_support_fraction = 0.0
        edge.qc.minimum_image_support = 0.0
        edge.qc.flags.add(QCFlag.IMAGE_SUPPORT)
        return

    image = np.asarray(frame, dtype=float)

    # Image-coordinate convention:
    # x increases along axis 1 and y increases along axis 0.
    gradient_x = sobel(image, axis=1, mode="nearest")
    gradient_y = sobel(image, axis=0, mode="nearest")

    theta = contour.theta
    radial_offsets = np.arange(
        -search_radius,
        search_radius + 1,
        dtype=float,
    )

    candidate_radii = (
        contour.r[:, np.newaxis]
        + radial_offsets[np.newaxis, :]
    )

    cos_theta = np.cos(theta)[:, np.newaxis]
    sin_theta = np.sin(theta)[:, np.newaxis]

    candidate_x = (
        contour.origin[0]
        + candidate_radii * cos_theta
    )
    candidate_y = (
        contour.origin[1]
        + candidate_radii * sin_theta
    )

    frame_height, frame_width = image.shape

    inside_image = (
        (candidate_x >= 0)
        & (candidate_x <= frame_width - 1)
        & (candidate_y >= 0)
        & (candidate_y <= frame_height - 1)
    )

    coordinates = np.vstack(
        (
            candidate_y.ravel(),
            candidate_x.ravel(),
        )
    )

    sampled_gradient_x = map_coordinates(
        gradient_x,
        coordinates,
        order=1,
        mode="nearest",
    ).reshape(candidate_x.shape)

    sampled_gradient_y = map_coordinates(
        gradient_y,
        coordinates,
        order=1,
        mode="nearest",
    ).reshape(candidate_y.shape)

    radial_gradient = np.abs(
        sampled_gradient_x * cos_theta
        + sampled_gradient_y * sin_theta
    )

    # Points outside the original image cannot provide support.
    radial_gradient = np.where(
        inside_image,
        radial_gradient,
        -np.inf,
    )

    edge_index = search_radius

    edge_gradient = radial_gradient[:, edge_index]
    strongest_local_gradient = np.max(
        radial_gradient,
        axis=1,
    )

    valid_edge_points = (
        inside_image[:, edge_index]
        & np.isfinite(edge_gradient)
        & np.isfinite(strongest_local_gradient)
        & (strongest_local_gradient > 0)
    )

    relative_support = np.zeros(
        contour.r.shape[0],
        dtype=float,
    )

    relative_support[valid_edge_points] = (
        edge_gradient[valid_edge_points]
        / strongest_local_gradient[valid_edge_points]
    )

    relative_support = np.clip(
        relative_support,
        0.0,
        1.0,
    )

    supported = relative_support >= threshold

    edge.qc.image_support_fraction = float(
        np.mean(supported)
    )
    edge.qc.minimum_image_support = float(
        np.min(relative_support)
    )

    # We require every angular position to be supported.
    if np.all(supported):
        edge.qc.flags.discard(QCFlag.IMAGE_SUPPORT)
    else:
        edge.qc.flags.add(QCFlag.IMAGE_SUPPORT)


def check_edge_populations(
    detections: list[object],
    bic_threshold: float = 10.0,
    max_minor_fraction: float = 0.25,
) -> EdgePopulationResult:
    """
    Identify a small anomalous population of edge detections.

    Only detections that passed all frame-level QC checks are included in the
    population analysis. Each detection is represented by its image-space
    center coordinates and median radius. The three features are robustly
    scaled before fitting one- and two-component Gaussian mixture models.

    If the two-component model improves BIC by at least `bic_threshold`, the
    detections are considered to contain two populations. The smaller
    population is automatically flagged as `QCFlag.POPULATION_OUTLIER` only
    when it contains no more than `max_minor_fraction` of the detections used
    for clustering.

    Parameters
    ----------
    detections : list
        Edge extraction results for all video frames. Extraction failures are
        ignored.
    bic_threshold : float, optional
        Minimum reduction in BIC required to accept the two-population model.
        Default is 10.
    max_minor_fraction : float, optional
        Maximum fraction of detections that may belong to the smaller
        population for that population to be automatically rejected. Must be
        greater than or equal to 0 and less than 0.5. Default is 0.25.

    Returns
    -------
    EdgePopulationResult
        Summary of the fitted population models and any automatic rejection.
    """
    if bic_threshold < 0:
        raise ValueError("bic_threshold must be non-negative.")

    if not 0 <= max_minor_fraction < 0.5:
        raise ValueError(
            "max_minor_fraction must be greater than or equal to 0 "
            "and less than 0.5."
        )

    usable_edges = [
        detection
        for detection in detections
        if _is_edge_detection(detection)
        and detection.qc.passed
    ]

    # At least two samples are needed to fit a two-component model, but a
    # two-component GMM fitted to only two observations is not useful for
    # distinguishing a trajectory population. Requiring four gives each
    # component the possibility of containing more than one observation.
    if len(usable_edges) < 4:
        for edge in usable_edges:
            edge.qc.population_label = 0
            edge.qc.population_probability = 1.0

        return EdgePopulationResult(
            bic_one_population=None,
            bic_two_populations=None,
            two_populations_detected=False,
            population_sizes=(len(usable_edges),),
            rejected_population=None,
        )

    features = np.array(
        [
            (
                edge.full_contour.origin[0],
                edge.full_contour.origin[1],
                edge.median_radius,
            )
            for edge in usable_edges
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            "Center and radius values used for population QC must be finite."
        )

    scaled_features = _robust_scale(features)

    one_population = GaussianMixture(
        n_components=1,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=0,
    )
    two_populations = GaussianMixture(
        n_components=2,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=0,
    )

    one_population.fit(scaled_features)
    two_populations.fit(scaled_features)

    bic_one = float(
        one_population.bic(scaled_features)
    )
    bic_two = float(
        two_populations.bic(scaled_features)
    )

    delta_bic = bic_one - bic_two

    if delta_bic < bic_threshold:
        for edge in usable_edges:
            edge.qc.population_label = 0
            edge.qc.population_probability = 1.0
            edge.qc.flags.discard(
                QCFlag.POPULATION_OUTLIER
            )

        return EdgePopulationResult(
            bic_one_population=bic_one,
            bic_two_populations=bic_two,
            two_populations_detected=False,
            population_sizes=(len(usable_edges),),
            rejected_population=None,
        )

    labels = two_populations.predict(
        scaled_features
    )
    probabilities = two_populations.predict_proba(
        scaled_features
    )

    assigned_probabilities = probabilities[
        np.arange(labels.shape[0]),
        labels,
    ]

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    for edge, label, probability in zip(
        usable_edges,
        labels,
        assigned_probabilities,
    ):
        edge.qc.population_label = int(label)
        edge.qc.population_probability = float(
            probability
        )
        edge.qc.flags.discard(
            QCFlag.POPULATION_OUTLIER
        )

    minor_index = int(np.argmin(counts))
    minor_label = int(
        unique_labels[minor_index]
    )
    minor_count = int(counts[minor_index])
    minor_fraction = (
        minor_count / len(usable_edges)
    )

    rejected_population = None

    if minor_fraction <= max_minor_fraction:
        rejected_population = minor_label

        for edge, label in zip(
            usable_edges,
            labels,
        ):
            if label == minor_label:
                edge.qc.flags.add(
                    QCFlag.POPULATION_OUTLIER
                )

    population_sizes = tuple(
        int(count)
        for count in counts
    )

    return EdgePopulationResult(
        bic_one_population=bic_one,
        bic_two_populations=bic_two,
        two_populations_detected=True,
        population_sizes=population_sizes,
        rejected_population=rejected_population,
    )


def _robust_scale(
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Robustly scale trajectory features before clustering.

    Each feature is centered on its median and divided by its interquartile
    range. If a feature has zero interquartile range, its standard deviation
    is used instead. Features with no variation are assigned a scale of one.

    Parameters
    ----------
    features : NDArray[np.float64]
        Two-dimensional array with observations along axis 0 and features
        along axis 1.

    Returns
    -------
    NDArray[np.float64]
        Robustly centered and scaled features.
    """
    medians = np.median(
        features,
        axis=0,
    )

    first_quartile = np.percentile(
        features,
        25,
        axis=0,
    )
    third_quartile = np.percentile(
        features,
        75,
        axis=0,
    )

    scale = third_quartile - first_quartile

    zero_iqr = scale == 0

    if np.any(zero_iqr):
        standard_deviation = np.std(
            features,
            axis=0,
        )
        scale[zero_iqr] = (
            standard_deviation[zero_iqr]
        )

    scale[scale == 0] = 1.0

    return (features - medians) / scale