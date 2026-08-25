#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic plots for VesEdge population quality control."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .models import EdgeDetection, EdgeResult


def save_population_histograms(
    detections: list[EdgeResult],
    path: str | Path,
) -> None:
    """Save histograms of the features used for population quality control.

    Only detections assigned a population label are shown. These are exactly
    the detections included in population fitting: extraction failures and
    detections rejected by preceding frame-level QC are omitted. If preceding
    QC leaves no usable detections, no figure is produced.

    Parameters
    ----------
    detections : list[EdgeResult]
        Ordered edge-extraction results after population QC has run.
    path : str or Path
        Destination for the PNG figure.

    Raises
    ------
    ValueError
        If usable detections exist but none has a population assignment,
        indicating population QC has not populated the requested diagnostics.
    """
    assigned_edges = [
        result
        for result in detections
        if isinstance(result, EdgeDetection)
        and result.qc.population_label is not None
    ]
    if not assigned_edges:
        usable_edges = [
            result
            for result in detections
            if isinstance(result, EdgeDetection)
            and result.qc.passed
        ]
        if not usable_edges:
            return
        raise ValueError(
            "Population QC must assign detections before histograms can be plotted."
        )

    features = _population_features(assigned_edges)
    labels = np.asarray(
        [edge.qc.population_label for edge in assigned_edges],
        dtype=int,
    )

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    _plot_feature_histogram(
        axis,
        features[:, 0],
        labels,
        "Median radius (pixels)",
    )

    figure.suptitle("Population QC radius distribution")
    output_path = Path(path).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_feature_histogram(axis, values, labels, feature_name) -> None:
    """Plot one population-QC feature grouped by fitted population."""
    bin_edges = np.histogram_bin_edges(values, bins="auto")
    colors = plt.get_cmap("tab10").colors
    for population_label in np.unique(labels):
        population_values = values[labels == population_label]
        axis.hist(
            population_values,
            bins=bin_edges,
            alpha=0.5,
            color=colors[int(population_label) % len(colors)],
            label=(
                f"Population {population_label} "
                f"(n={population_values.size})"
            ),
        )
    axis.set_xlabel(feature_name)
    axis.set_ylabel("Detections")
    axis.legend()


def _population_features(
    detections: list[EdgeDetection],
) -> NDArray[np.float64]:
    """Return the unscaled features used to fit population models."""
    return np.asarray(
        [
            [float(np.median(edge.analysis_contour.r))]
            for edge in detections
        ],
        dtype=float,
    )
