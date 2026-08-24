#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extracted VesEdge results, quality control, and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .checkpoint_io import load_checkpoint, save_checkpoint
from .config import EdgeExtractionConfig, EdgeQCConfig
from .edge_filtering import check_curvature, check_edge_populations
from .models import (
    CurvatureQCResult,
    EdgeDetection,
    EdgeDetectionFailure,
    EdgePopulationResult,
    EdgeQC,
    EdgeResult,
    QCFlag,
    VesicleQCResult,
)


@dataclass
class VesicleEdges:
    """Reusable edge-extraction results for one vesicle trajectory.

    Parameters
    ----------
    extraction_config : EdgeExtractionConfig
        Configuration used to construct the stored analysis contours.
    detections : list[EdgeResult]
        Ordered extraction result corresponding to each source video frame.
    """

    extraction_config: EdgeExtractionConfig
    detections: list[EdgeResult]
    qc_result: VesicleQCResult | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        """Validate extraction results stored on the object."""
        self._infer_frame_indices()
        self._validate_detection_lengths()

    @property
    def qc_config(self) -> EdgeQCConfig | None:
        """Return the configuration used for the most recent completed QC run."""
        if self.qc_result is None:
            return None
        return self.qc_result.config

    @property
    def successful_detections(self) -> list[EdgeDetection]:
        """Return all successfully extracted edge detections."""
        return [
            result
            for result in self.detections
            if isinstance(result, EdgeDetection)
        ]

    @property
    def accepted_detections(self) -> list[EdgeDetection]:
        """Return detections accepted by the most recent completed QC run.

        Raises
        ------
        ValueError
            If quality control has not yet completed on this object.
        """
        if self.qc_result is None:
            raise ValueError(
                "Quality control has not been run on these extracted edges."
            )
        return [
            detection
            for detection in self.successful_detections
            if detection.qc.passed
        ]

    @property
    def accepted_radii_microns(self) -> NDArray[np.float64]:
        """Return accepted analysis contours converted from pixels to microns."""
        accepted = self.accepted_detections
        if not accepted:
            raise ValueError(
                "No accepted edge detections are available."
            )
        return np.stack(
            [
                detection.analysis_contour.r
                for detection in accepted
            ]
        ) / self.extraction_config.pixels_per_micron

    def run_qc(
        self,
        qc_config: EdgeQCConfig | None = None,
    ) -> None:
        """Run enabled QC checks on stored detections.

        Existing QC state is cleared before every run. Supplying ``qc_config``
        replaces the most recently completed configuration; omitting it reuses
        that configuration. If a completed run rejects every detection, this
        method raises ``ValueError`` but retains the newly applied configuration,
        aggregate QC result, and per-detection QC flags for inspection.

        Raises
        ------
        ValueError
            If no QC configuration is available or no detection passes QC.
        """
        config = self.qc_config if qc_config is None else qc_config
        if config is None:
            raise ValueError(
                "A quality-control configuration is required before QC can run."
            )

        self._validate_detection_lengths()
        self._reset_qc()

        for detection in self.successful_detections:
            self._apply_frame_qc(detection, config)

        curvature_result = self._curvature_qc_result(config)
        population_result = self._apply_trajectory_qc(config)
        self.qc_result = VesicleQCResult(
            config=config,
            curvature=curvature_result,
            population=population_result,
        )
        self._validate_usable_detections()

    def _reset_qc(self) -> None:
        """Clear all previously derived quality-control state."""
        self.qc_result = None
        for detection in self.successful_detections:
            detection.qc = EdgeQC()

    @staticmethod
    def _apply_frame_qc(
        edge: EdgeDetection,
        config: EdgeQCConfig,
    ) -> None:
        """Apply enabled QC checks that operate on one detection."""
        if config.enable_curvature_qc:
            check_curvature(
                edge,
                threshold=config.curvature_threshold,
            )

    def _curvature_qc_result(
        self,
        config: EdgeQCConfig,
    ) -> CurvatureQCResult | None:
        """Summarize frame-level curvature QC for the completed run."""
        if not config.enable_curvature_qc:
            return None

        detections = self.successful_detections
        scores = tuple(
            float(detection.qc.curvature_score)
            if detection.qc.curvature_score is not None
            else float("nan")
            for detection in detections
        )
        rejected_count = sum(
            QCFlag.CURVATURE in detection.qc.flags
            for detection in detections
        )
        return CurvatureQCResult(
            scores=scores,
            rejected_count=rejected_count,
        )

    def _apply_trajectory_qc(
        self,
        config: EdgeQCConfig,
    ) -> EdgePopulationResult | None:
        """Apply enabled QC checks that operate across the trajectory."""
        if not config.enable_population_qc:
            return None
        return check_edge_populations(
            self.detections,
            bic_threshold=config.population_bic_threshold,
            max_minor_fraction=config.max_minor_population_fraction,
        )

    def _infer_frame_indices(self) -> None:
        """Infer missing frame indices and verify stored source-frame identity."""
        for expected_index, result in enumerate(self.detections):
            if result.frame_index is None:
                if isinstance(result, EdgeDetection):
                    result.frame_index = expected_index
                elif isinstance(result, EdgeDetectionFailure):
                    self.detections[expected_index] = replace(
                        result,
                        frame_index=expected_index,
                    )
                continue

            if result.frame_index != expected_index:
                raise ValueError(
                    "Edge result frame_index must match its source-frame "
                    f"position: expected {expected_index}, got "
                    f"{result.frame_index}."
                )

    def _validate_detection_lengths(self) -> None:
        """Verify successful detections have consistent analysis lengths."""
        unique_lengths = {
            detection.analysis_contour.r.shape[0]
            for detection in self.successful_detections
        }
        if not unique_lengths:
            raise ValueError(
                "Edge extraction produced no successful detections. "
                "Check the edge extractor implementation or input images."
            )
        if len(unique_lengths) > 1:
            raise ValueError(
                "Extracted edges have inconsistent numbers of angular samples."
            )

    def _validate_usable_detections(self) -> None:
        """Verify at least one successful detection passes current QC."""
        if not any(
            detection.qc.passed
            for detection in self.successful_detections
        ):
            raise ValueError(
                "Edge extraction produced detections, but no frames passed "
                "quality control."
            )

    def save_edge_to_npy(self, path: str | Path) -> None:
        """Save accepted analysis-contour radii in microns to ``.npy``."""
        np.save(
            Path(path).with_suffix(".npy"),
            self.accepted_radii_microns,
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Save reusable QC-independent extraction results to ``.npz``."""
        self._validate_detection_lengths()
        save_checkpoint(
            path,
            self.extraction_config,
            self.detections,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "VesicleEdges":
        """Restore extraction results from a VesEdge checkpoint without QC."""
        extraction_config, detections = load_checkpoint(path)
        return cls(
            extraction_config=extraction_config,
            detections=detections,
        )
