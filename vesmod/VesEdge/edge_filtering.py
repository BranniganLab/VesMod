#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quality-control algorithms for VesEdge edge detections."""

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture

from .models import (
    EdgeDetection,
    EdgePopulationResult,
    EdgeResult,
    QCFlag,
)
from .vesicle_video_utils import measure_wrapped_finite_second_difference

POPULATION_FEATURE_COUNT = 1
MIN_POPULATION_SAMPLE_COUNT = max(
    8,
    2 * (POPULATION_FEATURE_COUNT + 1),
)


def check_curvature(
    edge: EdgeDetection,
    threshold: float,
) -> None:
    """Check a detected edge for excessive local curvature.

    The edge fails curvature QC when the largest absolute wrapped finite
    second difference of its analysis contour exceeds ``threshold``.

    Raises
    ------
    ValueError
        If ``threshold`` is non-finite or negative.
    """
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
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

    if curvature_score > threshold:
        edge.qc.flags.add(QCFlag.CURVATURE)
    else:
        edge.qc.flags.discard(QCFlag.CURVATURE)


def check_edge_populations(
    detections: list[EdgeResult],
    bic_threshold: float,
    max_minor_fraction: float,
) -> EdgePopulationResult:
    """Identify a small anomalous radius population across frames.

    Only successful detections that passed all preceding QC checks are
    included. Each detection is represented by its median analysis-contour
    radius. Absolute center coordinates are excluded so camera panning and
    vesicle diffusion cannot create populations. Radius is robustly scaled
    before fitting one- and two-component Gaussian mixture models.

    Raises
    ------
    ValueError
        If ``bic_threshold`` is non-finite or negative, or if
        ``max_minor_fraction`` is non-finite or outside ``[0, 0.5)``.
    """
    _validate_population_parameters(
        bic_threshold,
        max_minor_fraction,
    )

    usable_edges = _get_usable_edges(detections)

    if len(usable_edges) < MIN_POPULATION_SAMPLE_COUNT:
        _assign_single_population(usable_edges)
        return _single_population_result(len(usable_edges))

    features = _population_features(usable_edges)
    scaled_features = _robust_scale(features)
    two_population_model, bic_values = _fit_population_models(
        scaled_features
    )

    if bic_values[0] - bic_values[1] < bic_threshold:
        _assign_single_population(usable_edges)
        return _single_population_result(
            len(usable_edges),
            *bic_values,
        )

    labels = two_population_model.predict(scaled_features)
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
        raise ValueError("bic_threshold must be finite.")
    if bic_threshold < 0:
        raise ValueError("bic_threshold must be non-negative.")
    if not np.isfinite(max_minor_fraction):
        raise ValueError("max_minor_fraction must be finite.")
    if not 0 <= max_minor_fraction < 0.5:
        raise ValueError(
            "max_minor_fraction must be greater than or "
            "equal to 0 and less than 0.5."
        )


def _get_usable_edges(
    detections: list[EdgeResult],
) -> list[EdgeDetection]:
    """Return successful detections that passed preceding QC checks."""
    return [
        result
        for result in detections
        if isinstance(result, EdgeDetection)
        and result.qc.passed
    ]


def _population_features(
    usable_edges: list[EdgeDetection],
) -> NDArray[np.float64]:
    """Build and validate radius-only features for population QC."""
    features = np.asarray(
        [
            [float(np.median(edge.analysis_contour.r))]
            for edge in usable_edges
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            "Radius values used for population QC must be finite."
        )

    return features


def _fit_population_models(
    scaled_features: NDArray[np.float64],
) -> tuple[GaussianMixture, tuple[float, float]]:
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

    one_population_model.fit(scaled_features)
    two_population_model.fit(scaled_features)

    bic_one = float(one_population_model.bic(scaled_features))
    bic_two = float(two_population_model.bic(scaled_features))

    return two_population_model, (bic_one, bic_two)


def _assign_single_population(
    usable_edges: list[EdgeDetection],
) -> None:
    """Assign detections to one population."""
    for edge in usable_edges:
        edge.qc.population_label = 0
        edge.qc.population_probability = 1.0


def _single_population_result(
    population_size: int,
    bic_one: float | None = None,
    bic_two: float | None = None,
) -> EdgePopulationResult:
    """Build a result describing a single detected population."""
    return EdgePopulationResult(
        bic_one_population=bic_one,
        bic_two_populations=bic_two,
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
        edge.qc.population_probability = float(probability)

    minor_index = int(np.argmin(counts))
    minor_label = int(unique_labels[minor_index])
    minor_fraction = int(counts[minor_index]) / len(usable_edges)

    rejected_population = _reject_minor_population(
        usable_edges,
        labels,
        minor_label,
        minor_fraction,
        max_minor_fraction,
    )

    return EdgePopulationResult(
        bic_one_population=bic_values[0],
        bic_two_populations=bic_values[1],
        two_populations_detected=True,
        population_sizes=tuple(int(count) for count in counts),
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
            edge.qc.flags.add(QCFlag.POPULATION_OUTLIER)

    return minor_label


def _robust_scale(
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Robustly center and scale features before population clustering."""
    medians = np.median(features, axis=0)
    first_quartile = np.percentile(features, 25, axis=0)
    third_quartile = np.percentile(features, 75, axis=0)
    scale = third_quartile - first_quartile

    zero_iqr = scale == 0
    if np.any(zero_iqr):
        standard_deviation = np.std(features, axis=0)
        scale[zero_iqr] = standard_deviation[zero_iqr]

    scale[scale == 0] = 1.0
    return (features - medians) / scale
