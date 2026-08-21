#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extracted VesEdge results, quality control, and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import EdgeExtractionConfig, EdgeQCConfig
from .edge_filtering import (
    check_curvature,
    check_edge_populations,
)
from .models import (
    EdgeDetection,
    EdgeDetectionFailure,
    EdgePopulationResult,
    EdgeQC,
    EdgeResult,
    ImageContour,
)

CHECKPOINT_VERSION = 2
_DETECTION_CODE = 1
_FAILURE_CODE = 0


@dataclass
class VesicleEdges:
    """Edge-extraction results for one vesicle trajectory.

    This object represents the reusable output of edge extraction. It owns the
    ordered per-frame extraction results and supports quality control,
    checkpoint persistence, and export of accepted radii for EdgeMod. Raw image
    frames are intentionally not stored here.

    Parameters
    ----------
    extraction_config : EdgeExtractionConfig
        Configuration used to construct the stored analysis contours.
    detections : list[EdgeResult]
        Ordered extraction result corresponding to each source video frame.
    qc_config : EdgeQCConfig | None, optional
        Configuration used for the most recent QC run. None means QC has not
        been run on the current object.
    """

    extraction_config: EdgeExtractionConfig
    detections: list[EdgeResult]
    qc_config: EdgeQCConfig | None = None
    population_result: EdgePopulationResult | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        """Validate extraction results stored on the object."""
        self._validate_detection_lengths()

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
        """Return detections accepted by the most recent QC run.

        Raises
        ------
        ValueError
            If quality control has not yet been run.
        """
        if self.qc_config is None:
            raise ValueError(
                "Quality control has not been run on these extracted edges."
            )
        return [
            detection
            for detection in self.successful_detections
            if detection.accepted
        ]

    def run_qc(
        self,
        qc_config: EdgeQCConfig | None = None,
    ) -> None:
        """Run enabled quality-control checks on stored detections.

        Existing QC state is cleared before every run. Supplying ``qc_config``
        replaces the current configuration; omitting it reuses the most recent
        configuration.

        Parameters
        ----------
        qc_config : EdgeQCConfig | None, optional
            QC configuration to apply.

        Raises
        ------
        ValueError
            If no QC configuration is available, if the stored detections are
            invalid, or if no successful detection passes quality control.
        """
        if qc_config is not None:
            self.qc_config = qc_config
        if self.qc_config is None:
            raise ValueError(
                "A quality-control configuration is required before QC can run."
            )

        self._validate_detection_lengths()
        self._reset_qc()

        for detection in self.successful_detections:
            self._apply_frame_qc(detection)

        self._apply_trajectory_qc()
        self._validate_usable_detections()

    def _reset_qc(self) -> None:
        """Clear all previously derived quality-control state."""
        self.population_result = None
        for detection in self.successful_detections:
            detection.qc = EdgeQC()

    def _apply_frame_qc(self, edge: EdgeDetection) -> None:
        """Apply enabled QC checks that operate on one detection."""
        if self.qc_config is None:
            raise RuntimeError("QC configuration is unavailable.")
        if self.qc_config.enable_curvature_qc:
            check_curvature(
                edge,
                threshold=self.qc_config.curvature_threshold,
            )

    def _apply_trajectory_qc(self) -> None:
        """Apply enabled QC checks that operate across the trajectory."""
        self.population_result = None
        if self.qc_config is None:
            raise RuntimeError("QC configuration is unavailable.")
        if self.qc_config.enable_population_qc:
            self.population_result = check_edge_populations(
                self.detections,
                bic_threshold=self.qc_config.population_bic_threshold,
                max_minor_fraction=(
                    self.qc_config.max_minor_population_fraction
                ),
            )

    def _validate_detection_lengths(self) -> None:
        """Verify successful detections have consistent analysis lengths."""
        unique_lengths = {
            detection.radii_microns.shape[0]
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
            detection.accepted
            for detection in self.successful_detections
        ):
            raise ValueError(
                "Edge extraction produced detections, but no frames passed "
                "quality control."
            )

    def save_edge_to_npy(self, path: str | Path) -> None:
        """Save radii from detections accepted by the current QC run.

        Parameters
        ----------
        path : str | pathlib.Path
            Output path. The suffix is replaced with ``.npy``.

        Raises
        ------
        ValueError
            If QC has not been run or no detection is currently accepted.
        """
        accepted = self.accepted_detections
        if not accepted:
            raise ValueError(
                "Cannot save edges: no accepted edge detections are available."
            )
        np.save(
            Path(path).with_suffix(".npy"),
            np.stack(
                [edge.radii_microns for edge in accepted]
            ),
        )

    def save_checkpoint(self, path: str | Path) -> None:
        """Save the reusable, unfiltered result of edge extraction.

        QC settings and QC results are intentionally excluded. A checkpoint
        represents extraction state that may be evaluated under any later QC
        configuration.

        Parameters
        ----------
        path : str | pathlib.Path
            Output path. The suffix is replaced with ``.npz``.
        """
        self._validate_detection_lengths()
        successful = self.successful_detections
        result_types = np.asarray(
            [
                _DETECTION_CODE
                if isinstance(result, EdgeDetection)
                else _FAILURE_CODE
                for result in self.detections
            ],
            dtype=np.uint8,
        )
        failure_errors = np.asarray(
            [
                result.error
                for result in self.detections
                if isinstance(result, EdgeDetectionFailure)
            ],
            dtype=str,
        )
        origins = np.asarray(
            [edge.full_contour.origin for edge in successful],
            dtype=float,
        )
        analysis_radii_pixels = np.stack(
            [edge.analysis_contour.r for edge in successful]
        )
        radii_microns = np.stack(
            [edge.radii_microns for edge in successful]
        )
        full_radii_values, full_radii_offsets = self._flatten_full_radii(
            successful
        )
        n_angular_samples = (
            -1
            if self.extraction_config.n_angular_samples is None
            else self.extraction_config.n_angular_samples
        )

        np.savez(
            Path(path).with_suffix(".npz"),
            checkpoint_version=np.asarray(CHECKPOINT_VERSION),
            pixels_per_micron=np.asarray(
                self.extraction_config.pixels_per_micron,
                dtype=float,
            ),
            n_angular_samples=np.asarray(
                n_angular_samples,
                dtype=np.int64,
            ),
            result_types=result_types,
            failure_errors=failure_errors,
            origins=origins,
            full_radii_values=full_radii_values,
            full_radii_offsets=full_radii_offsets,
            analysis_radii_pixels=analysis_radii_pixels,
            radii_microns=radii_microns,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "VesicleEdges":
        """Restore extraction results from a VesEdge checkpoint.

        Loading does not run quality control. Call :meth:`run_qc` with the
        desired configuration before exporting accepted edges or constructing a
        Spectrum from this object.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )
        if checkpoint_path.suffix != ".npz":
            raise ValueError("VesEdge checkpoints must end in .npz.")

        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            saved_data = {
                key: checkpoint[key].copy()
                for key in checkpoint.files
            }

        cls._validate_checkpoint_keys(saved_data)
        version = int(saved_data["checkpoint_version"])
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported VesEdge checkpoint version: "
                f"{version}."
            )

        return cls(
            extraction_config=cls._extraction_config_from_checkpoint(
                saved_data
            ),
            detections=cls._detections_from_checkpoint(saved_data),
        )

    @staticmethod
    def _flatten_full_radii(
        detections: list[EdgeDetection],
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Flatten possibly variable-length native contours with offsets."""
        lengths = np.asarray(
            [edge.full_contour.r.shape[0] for edge in detections],
            dtype=np.int64,
        )
        offsets = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum(lengths, dtype=np.int64),
            )
        )
        values = np.concatenate(
            [edge.full_contour.r for edge in detections]
        ).astype(float, copy=False)
        return values, offsets

    @staticmethod
    def _validate_checkpoint_keys(
        checkpoint: dict[str, np.ndarray],
    ) -> None:
        """Verify that a checkpoint contains all required extraction fields."""
        required_keys = {
            "checkpoint_version",
            "pixels_per_micron",
            "n_angular_samples",
            "result_types",
            "failure_errors",
            "origins",
            "full_radii_values",
            "full_radii_offsets",
            "analysis_radii_pixels",
            "radii_microns",
        }
        missing_keys = required_keys - checkpoint.keys()
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(
                "VesEdge checkpoint is missing required field(s): "
                f"{missing}."
            )

    @staticmethod
    def _extraction_config_from_checkpoint(
        checkpoint: dict[str, np.ndarray],
    ) -> EdgeExtractionConfig:
        """Reconstruct extraction settings stored in a checkpoint."""
        stored_samples = int(checkpoint["n_angular_samples"])
        return EdgeExtractionConfig(
            pixels_per_micron=float(checkpoint["pixels_per_micron"]),
            n_angular_samples=(
                None
                if stored_samples == -1
                else stored_samples
            ),
        )

    @classmethod
    def _detections_from_checkpoint(
        cls,
        checkpoint: dict[str, np.ndarray],
    ) -> list[EdgeResult]:
        """Reconstruct ordered extraction results from checkpoint arrays."""
        result_types = checkpoint["result_types"]
        success_count = int(
            np.count_nonzero(result_types == _DETECTION_CODE)
        )
        failure_count = int(
            np.count_nonzero(result_types == _FAILURE_CODE)
        )
        cls._validate_checkpoint_shapes(
            checkpoint,
            success_count,
            failure_count,
        )

        successful = cls._successful_detections_from_checkpoint(
            checkpoint["origins"],
            checkpoint["analysis_radii_pixels"],
            checkpoint["radii_microns"],
            checkpoint["full_radii_values"],
            checkpoint["full_radii_offsets"],
        )
        return cls._merge_checkpoint_results(
            result_types,
            successful,
            checkpoint["failure_errors"],
        )

    @staticmethod
    def _validate_checkpoint_shapes(
        checkpoint: dict[str, np.ndarray],
        success_count: int,
        failure_count: int,
    ) -> None:
        """Validate checkpoint array counts and shapes."""
        result_types = checkpoint["result_types"]
        failure_errors = checkpoint["failure_errors"]
        origins = checkpoint["origins"]
        analysis_radii = checkpoint["analysis_radii_pixels"]
        radii_microns = checkpoint["radii_microns"]
        full_offsets = checkpoint["full_radii_offsets"]
        full_values = checkpoint["full_radii_values"]

        valid_codes = np.isin(
            result_types,
            [_DETECTION_CODE, _FAILURE_CODE],
        )
        if result_types.ndim != 1 or not np.all(valid_codes):
            raise ValueError(
                "VesEdge checkpoint contains invalid result types."
            )
        if failure_errors.shape != (failure_count,):
            raise ValueError(
                "VesEdge checkpoint failure metadata are inconsistent."
            )
        if origins.shape != (success_count, 2):
            raise ValueError(
                "VesEdge checkpoint origin metadata are inconsistent."
            )
        if (
            analysis_radii.ndim != 2
            or analysis_radii.shape[0] != success_count
            or radii_microns.shape != analysis_radii.shape
        ):
            raise ValueError(
                "VesEdge checkpoint analysis contours are inconsistent."
            )
        if full_offsets.shape != (success_count + 1,):
            raise ValueError(
                "VesEdge checkpoint native-contour offsets are inconsistent."
            )
        if (
            full_offsets[0] != 0
            or full_offsets[-1] != full_values.shape[0]
            or np.any(np.diff(full_offsets) <= 0)
        ):
            raise ValueError(
                "VesEdge checkpoint native contours are inconsistent."
            )

    @staticmethod
    def _successful_detections_from_checkpoint(
        origins: np.ndarray,
        analysis_radii: np.ndarray,
        radii_microns: np.ndarray,
        full_values: np.ndarray,
        full_offsets: np.ndarray,
    ) -> list[EdgeDetection]:
        """Reconstruct successful detections from checkpoint arrays."""
        detections = []
        for index, origin_values in enumerate(origins):
            start = int(full_offsets[index])
            stop = int(full_offsets[index + 1])
            origin = (
                float(origin_values[0]),
                float(origin_values[1]),
            )
            detections.append(
                EdgeDetection(
                    ImageContour(
                        origin,
                        full_values[start:stop].copy(),
                    ),
                    ImageContour(
                        origin,
                        analysis_radii[index].copy(),
                    ),
                    radii_microns[index].copy(),
                )
            )
        return detections

    @staticmethod
    def _merge_checkpoint_results(
        result_types: np.ndarray,
        successful: list[EdgeDetection],
        failure_errors: np.ndarray,
    ) -> list[EdgeResult]:
        """Restore original frame ordering of successes and failures."""
        detections: list[EdgeResult] = []
        success_index = 0
        failure_index = 0
        for result_type in result_types:
            if int(result_type) == _DETECTION_CODE:
                detections.append(successful[success_index])
                success_index += 1
            else:
                detections.append(
                    EdgeDetectionFailure(
                        str(failure_errors[failure_index])
                    )
                )
                failure_index += 1
        return detections
