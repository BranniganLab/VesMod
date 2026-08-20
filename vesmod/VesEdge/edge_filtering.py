#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality-control routines for VesEdge edge detections.

Contains configuration and algorithms for evaluating successfully extracted
vesicle edges. Frame-level QC examines individual detections, while
trajectory-level QC compares detections across a video.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture

from .models import (
    EdgeDetection,
    EdgeResult,
    QCFlag,
    FRAME_QC_FLAGS,
)
from .vesicle_video_utils import measure_wrapped_finite_second_difference

POPULATION_FEATURE_COUNT = 3
MIN_POPULATION_SAMPLE_COUNT = 2 * (POPULATION_FEATURE_COUNT + 1)


@dataclass(frozen=True)
class EdgeQCConfig:
    """
    Configuration parameters for VesEdge quality-control checks.

    Attributes
    ----------
    curvature_threshold : float
        Maximum allowed absolute wrapped finite second difference of an
        analysis contour.
    population_bic_threshold : float
        Minimum improvement in Bayesian information criterion (BIC) required
        for a two-population Gaussian mixture model to be preferred over a
        one-population model.
    max_minor_population_fraction : float
        Maximum fraction of otherwise accepted detections that may belong to
        the smaller population for that population to be automatically
        rejected.
    enable_curvature_qc : bool
        Whether frame-level curvature QC runs. Default is True.
    enable_population_qc : bool
        Whether trajectory-level population QC runs. Default is True.
    """

    curvature_threshold: float
    population_bic_threshold: float
    max_minor_population_fraction: float
    enable_curvature_qc: bool = True
    enable_population_qc: bool = True

    def __post_init__(self) -> None:
        """
        Validate quality-control configuration parameters.

        Raises
        ------
        ValueError
            If any numeric parameter lies outside its allowed range.
        """
        if not np.isfinite(self.curvature_threshold):
            raise ValueError("curvature_threshold must be finite.")
        if self.curvature_threshold < 0:
            raise ValueError(
                "curvature_threshold must be non-negative."
            )

        if not np.isfinite(self.population_bic_threshold):
            raise ValueError(
                "population_bic_threshold must be finite."
            )
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


@dataclass(frozen=True)
class EdgePopulationResult:
    """
    Results from trajectory-level center/radius population analysis.

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

        return (
            self.bic_one_population
            - self.bic_two_populations
        )


def check_curvature(
    edge: EdgeDetection,
    threshold: float,
) -> None:
    """
    Check a detected edge for excessive local curvature.

    Calculates the wrapped finite second difference of the radial profile in
    `edge.analysis_contour`. The largest absolute value is recorded as the
    curvature score. The edge fails curvature QC if this score is greater
    than or equal to `threshold`.

    Parameters
    ----------
    edge : EdgeDetection
        Edge detection to evaluate.
    threshold : float
        Maximum allowed absolute wrapped finite second difference.

    Raises
    ------
    ValueError
        If ``threshold`` is non-finite or negative.

    Returns
    -------
    None
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    finite_second_difference = (
        measure_wrapped_finite_second_difference(
            edge.analysis_contour.r
        )
    )

    if not np.all(
        np.isfinite(finite_second_difference)
    ):
        edge.qc.curvature_score = np.nan
        edge.qc.flags.add(QCFlag.CURVATURE)
        return

    curvature_score = float(
        np.max(
            np.abs(finite_second_difference)
        )
    )

    edge.qc.curvature_score = curvature_score

    if curvature_score >= threshold:
        edge.qc.flags.add(QCFlag.CURVATURE)
    else:
        edge.qc.flags.discard(QCFlag.CURVATURE)


def check_edge_populations(
    detections: list[EdgeResult],
    bic_threshold: float,
    max_minor_fraction: float,
) -> EdgePopulationResult:
    """
    Identify a small anomalous center/radius population across video frames.

    Only successful detections that passed all frame-level QC checks are
    included. Each detection is represented by its image-space center
    coordinates and median physical radius.

    The features are robustly scaled before fitting one- and two-component
    Gaussian mixture models. The two-population model is accepted when its
    BIC improves upon the one-population model by at least `bic_threshold`.

    If two populations are detected, the smaller population is rejected only
    if it contains no more than `max_minor_fraction` of the detections used
    for clustering.

    Parameters
    ----------
    detections : list[EdgeResult]
        Edge-extraction results for all video frames. Failed extractions and
        detections that already failed frame-level QC are excluded.
    bic_threshold : float
        Minimum reduction in BIC required to accept the two-population model.
    max_minor_fraction : float
        Maximum fraction of detections that may belong to the smaller
        population for that population to be automatically rejected. Must be
        greater than or equal to 0 and less than 0.5.

    Raises
    ------
    ValueError
        If ``bic_threshold`` is non-finite or negative, or if
        ``max_minor_fraction`` is non-finite or outside the interval
        ``[0, 0.5)``.

    Returns
    -------
    EdgePopulationResult
        Results of the one- versus two-population comparison.
    """
    _validate_population_parameters(
        bic_threshold,
        max_minor_fraction,
    )

    usable_edges = _get_usable_edges(
        detections
    )

    for edge in usable_edges:
        edge.qc.flags.discard(
            QCFlag.POPULATION_OUTLIER
        )

    if len(usable_edges) < MIN_POPULATION_SAMPLE_COUNT:
        _assign_single_population(
            usable_edges,
            clear_outlier_flag=False,
        )
        return _single_population_result(
            len(usable_edges)
        )

    features = _population_features(usable_edges)
    scaled_features = _robust_scale(features)
    (
        two_population_model,
        bic_values,
    ) = _fit_population_models(
        scaled_features
    )

    if (
        bic_values[0] - bic_values[1]
        < bic_threshold
    ):
        _assign_single_population(
            usable_edges,
            clear_outlier_flag=True,
        )
        return _single_population_result(
            len(usable_edges),
            *bic_values,
        )

    labels = two_population_model.predict(
        scaled_features
    )
    probabilities = two_population_model.predict_proba(
        scaled_features
    )

    return _apply_two_population_results(
        usable_edges,
        labels,
        probabilities,
        bic_values,
        max_minor_fraction,
    )


def _validate_population_parameters(
    bic_threshold: float,
    max_minor_fraction: float,
) -> None:
    """Validate trajectory-level population-QC parameters."""
    if not np.isfinite(bic_threshold):
        raise ValueError(
            "bic_threshold must be finite."
        )
    if bic_threshold < 0:
        raise ValueError(
            "bic_threshold must be non-negative."
        )
    if not np.isfinite(max_minor_fraction):
        raise ValueError(
            "max_minor_fraction must be finite."
        )
    if not 0 <= max_minor_fraction < 0.5:
        raise ValueError(
            "max_minor_fraction must be greater than or "
            "equal to 0 and less than 0.5."
        )


def _get_usable_edges(
    detections: list[EdgeResult],
) -> list[EdgeDetection]:
    """Return detections eligible for trajectory-level population QC."""
    return [
        result
        for result in detections
        if isinstance(result, EdgeDetection)
        and _passes_frame_qc(result)
    ]


def _passes_frame_qc(
    edge: EdgeDetection,
) -> bool:
    """Return whether an edge passed all frame-level QC checks."""
    return not (
        edge.qc.flags
        & FRAME_QC_FLAGS
    )


def _population_features(
    usable_edges: list[EdgeDetection],
) -> NDArray[np.float64]:
    """Build and validate center/radius features for population QC."""
    features = np.asarray(
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
            "Center and radius values used for population QC "
            "must be finite."
        )

    return features


def _fit_population_models(
    scaled_features: NDArray[np.float64],
) -> tuple[
    GaussianMixture,
    tuple[float, float],
]:
    """Fit one- and two-population Gaussian mixture models."""
    one_population_model = GaussianMixture(
        n_components=1,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=0,
    )
    two_population_model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=0,
    )

    one_population_model.fit(
        scaled_features
    )
    two_population_model.fit(
        scaled_features
    )

    bic_one = float(
        one_population_model.bic(
            scaled_features
        )
    )
    bic_two = float(
        two_population_model.bic(
            scaled_features
        )
    )

    return (
        two_population_model,
        (bic_one, bic_two),
    )


def _assign_single_population(
    usable_edges: list[EdgeDetection],
    *,
    clear_outlier_flag: bool,
) -> None:
    """Assign detections to one population."""
    for edge in usable_edges:
        edge.qc.population_label = 0
        edge.qc.population_probability = 1.0
        if clear_outlier_flag:
            edge.qc.flags.discard(
                QCFlag.POPULATION_OUTLIER
            )


def _single_population_result(
    population_size: int,
    bic_one: float | None = None,
    bic_two: float | None = None,
) -> EdgePopulationResult:
    """Build a result describing a single detected population."""
    return EdgePopulationResult(
        bic_one_population=bic_values[0],
        bic_two_populations=bic_values[1],
        two_populations_detected=False,
        population_sizes=(population_size,),
        rejected_population=None,
    )


def _apply_two_population_results(
    usable_edges: list[EdgeDetection],
    labels: NDArray[np.integer],
    probabilities: NDArray[np.float64],
    bic_values: tuple[float, float],
    max_minor_fraction: float,
) -> EdgePopulationResult:
    """Record two-population assignments and reject a small minor population."""
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
        strict=True,
    ):
        edge.qc.population_label = int(label)
        edge.qc.population_probability = float(
            probability
        )
        edge.qc.flags.discard(
            QCFlag.POPULATION_OUTLIER
        )

    minor_index = int(np.argmin(counts))
    minor_label = int(unique_labels[minor_index])
    minor_fraction = (
        int(counts[minor_index])
        / len(usable_edges)
    )

    rejected_population = _reject_minor_population(
        usable_edges,
        labels,
        minor_label,
        minor_fraction,
        max_minor_fraction,
    )

    return EdgePopulationResult(
        bic_one_population=bic_one,
        bic_two_populations=bic_two,
        two_populations_detected=True,
        population_sizes=tuple(
            int(count)
            for count in counts
        ),
        rejected_population=rejected_population,
    )


def _reject_minor_population(
    usable_edges: list[EdgeDetection],
    labels: NDArray[np.integer],
    minor_label: int,
    minor_fraction: float,
    max_minor_fraction: float,
) -> int | None:
    """Flag the minor population when it falls below the rejection threshold."""
    if minor_fraction > max_minor_fraction:
        return None

    for edge, label in zip(
        usable_edges,
        labels,
        strict=True,
    ):
        if label == minor_label:
            edge.qc.flags.add(
                QCFlag.POPULATION_OUTLIER
            )

    return minor_label


def _robust_scale(
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Robustly scale features before population clustering.

    Each feature is centered on its median and divided by its interquartile
    range. If the interquartile range is zero, the standard deviation is used
    instead. Features with no variation are assigned a scale of one.

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

    scale = (
        third_quartile
        - first_quartile
    )

    zero_iqr = scale == 0

    if np.any(zero_iqr):
        standard_deviation = np.std(
            features,
            axis=0,
        )

        scale[zero_iqr] = (
            standard_deviation[
                zero_iqr
            ]
        )

    scale[scale == 0] = 1.0

    return (
        features - medians
    ) / scale
