#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extracted VesEdge results, quality control, and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .io.checkpoint_io import load_checkpoint, save_checkpoint
from .extraction.config import EdgeExtractionConfig
from .io.frame_source import OnDemandFrameSequence
from .models import (
    EdgeDetection,
    EdgeDetectionFailure,
    EdgeResult,
    VesicleQCResult,
)
from .qc.qc_runner import run_configured_qc_checks
from .qc.qc_config import VesicleQCConfig


@dataclass
class VesicleEdges:
    """Reusable edge-extraction results for one vesicle trajectory.

    Parameters
    ----------
    extraction_config : EdgeExtractionConfig
        Configuration used to construct the stored analysis contours.
    detections : list[EdgeResult]
        Ordered extraction result corresponding to each source video frame.
    source_path : str | Path | None
        Path to the original source video, when known.
    """

    extraction_config: EdgeExtractionConfig
    detections: list[EdgeResult]
    source_path: str | Path | None = None
    qc_result: VesicleQCResult | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        """Validate extraction results stored on the object."""
        if self.source_path is not None:
            self.source_path = Path(self.source_path)
        self._infer_frame_indices()
        self._validate_detection_lengths()

    @property
    def qc_config(self) -> VesicleQCConfig | None:
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
        accepted = set(self.qc_result.accepted_frame_indices)
        return [
            detection
            for detection in self.successful_detections
            if detection.frame_index in accepted
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
        qc_config: VesicleQCConfig | None = None,
        frames: OnDemandFrameSequence | NDArray[np.number] | None = None,
    ) -> None:
        """Run enabled QC checks on stored detections.

        Supplying ``qc_config`` replaces the most recently completed configuration; omitting it reuses
        that configuration. If a completed run rejects every detection, this
        method raises ``ValueError`` but retains the newly applied configuration
        and aggregate QC result for inspection. If a check raises before
        producing a completed result, the previous result remains available.

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
        result = run_configured_qc_checks(self.successful_detections, config, frames)
        self.qc_result = result
        self._validate_usable_detections()

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
        if self.qc_result is not None and not self.qc_result.passed:
            raise ValueError(
                "Vesicle trajectory failed quality control."
            )
        if not self.qc_result.accepted_frame_indices:
            raise ValueError(
                "Edge extraction produced detections, but no frames passed "
                "quality control."
            )

    def export_accepted_radii(self, path: str | Path) -> None:
        """Export all QC-accepted analysis-contour radii in microns to ``.npy``."""
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
            source_path=self.source_path,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "VesicleEdges":
        """Restore extraction results from a VesEdge checkpoint without QC."""
        extraction_config, detections, source_path = load_checkpoint(path)
        return cls(
            extraction_config=extraction_config,
            detections=detections,
            source_path=source_path,
        )
