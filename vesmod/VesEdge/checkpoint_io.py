#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serialization helpers for VesEdge extraction checkpoints."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import EdgeExtractionConfig
from .models import (
    EdgeDetection,
    EdgeDetectionFailure,
    EdgeResult,
    ImageContour,
)

_DETECTION_CODE = 1
_FAILURE_CODE = 0


def save_checkpoint(
    path: str | Path,
    extraction_config: EdgeExtractionConfig,
    detections: list[EdgeResult],
    source_path: str | Path | None = None,
) -> None:
    """Save QC-independent extraction results to a ``.npz`` file."""
    successful = [
        result
        for result in detections
        if isinstance(result, EdgeDetection)
    ]
    if not successful:
        raise ValueError(
            "Cannot save a checkpoint with no successful detections."
        )
    if any(
        edge.full_contour.origin != edge.analysis_contour.origin
        for edge in successful
    ):
        raise ValueError(
            "Cannot save a checkpoint when full and analysis contour origins "
            "differ."
        )

    raw_frame_indices = [result.frame_index for result in detections]
    if any(
        isinstance(frame_index, (bool, np.bool_))
        or not isinstance(frame_index, (int, np.integer))
        for frame_index in raw_frame_indices
    ):
        raise ValueError(
            "Cannot save a checkpoint with missing or inconsistent frame indices."
        )
    frame_indices = np.asarray(raw_frame_indices, dtype=np.int64)
    expected_indices = np.arange(len(detections), dtype=np.int64)
    if not np.array_equal(frame_indices, expected_indices):
        raise ValueError(
            "Cannot save a checkpoint with missing or inconsistent frame indices."
        )

    result_types = np.asarray(
        [
            _DETECTION_CODE
            if isinstance(result, EdgeDetection)
            else _FAILURE_CODE
            for result in detections
        ],
        dtype=np.uint8,
    )
    failure_errors = np.asarray(
        [
            result.error
            for result in detections
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
    full_radii_values, full_radii_offsets = _flatten_full_radii(successful)
    n_angular_samples = (
        -1
        if extraction_config.n_angular_samples is None
        else extraction_config.n_angular_samples
    )

    checkpoint_data = {
        "pixels_per_micron": np.asarray(
            extraction_config.pixels_per_micron,
            dtype=float,
        ),
        "n_angular_samples": np.asarray(
            n_angular_samples,
            dtype=np.int64,
        ),
        "calibration_source": np.asarray(
            extraction_config.calibration_source,
        ),
        "frame_indices": frame_indices,
        "result_types": result_types,
        "failure_errors": failure_errors,
        "origins": origins,
        "full_radii_values": full_radii_values,
        "full_radii_offsets": full_radii_offsets,
        "analysis_radii_pixels": analysis_radii_pixels,
    }
    if source_path is not None:
        checkpoint_data["source_path"] = np.asarray(str(source_path))

    np.savez(
        Path(path).with_suffix(".npz"),
        **checkpoint_data,
    )


def load_checkpoint(
    path: str | Path,
) -> tuple[EdgeExtractionConfig, list[EdgeResult], Path | None]:
    """Load QC-independent extraction results from a VesEdge checkpoint."""
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

    _validate_checkpoint_keys(saved_data)

    extraction_config = _extraction_config_from_checkpoint(saved_data)
    detections = _detections_from_checkpoint(saved_data)
    source_path = _source_path_from_checkpoint(saved_data)
    return extraction_config, detections, source_path


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


def _validate_checkpoint_keys(
    checkpoint: dict[str, np.ndarray],
) -> None:
    """Verify that a checkpoint contains all required extraction fields."""
    required_keys = {
        "pixels_per_micron",
        "n_angular_samples",
        "result_types",
        "failure_errors",
        "origins",
        "full_radii_values",
        "full_radii_offsets",
        "analysis_radii_pixels",
    }

    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(
            "VesEdge checkpoint is missing required field(s): "
            f"{missing}."
        )


def _extraction_config_from_checkpoint(
    checkpoint: dict[str, np.ndarray],
) -> EdgeExtractionConfig:
    """Reconstruct extraction settings stored in a checkpoint."""
    stored_samples = int(checkpoint["n_angular_samples"])
    calibration_source = (
        str(checkpoint["calibration_source"].item())
        if "calibration_source" in checkpoint
        else "unspecified"
    )
    return EdgeExtractionConfig(
        pixels_per_micron=float(checkpoint["pixels_per_micron"]),
        n_angular_samples=(
            None if stored_samples == -1 else stored_samples
        ),
        calibration_source=calibration_source,
    )


def _source_path_from_checkpoint(
    checkpoint: dict[str, np.ndarray],
) -> Path | None:
    """Return persisted source-video provenance when present."""
    if "source_path" not in checkpoint:
        return None
    return Path(str(checkpoint["source_path"].item()))


def _detections_from_checkpoint(
    checkpoint: dict[str, np.ndarray],
) -> list[EdgeResult]:
    """Reconstruct ordered extraction results from checkpoint arrays."""
    result_types = checkpoint["result_types"]
    frame_indices = checkpoint.get(
        "frame_indices",
        np.arange(result_types.shape[0], dtype=np.int64),
    )
    success_count = int(
        np.count_nonzero(result_types == _DETECTION_CODE)
    )
    failure_count = int(
        np.count_nonzero(result_types == _FAILURE_CODE)
    )
    _validate_checkpoint_shapes(
        checkpoint,
        frame_indices,
        success_count,
        failure_count,
    )

    successful = _successful_detections_from_checkpoint(
        checkpoint["origins"],
        checkpoint["analysis_radii_pixels"],
        checkpoint["full_radii_values"],
        checkpoint["full_radii_offsets"],
        frame_indices[result_types == _DETECTION_CODE],
    )
    return _merge_checkpoint_results(
        result_types,
        frame_indices,
        successful,
        checkpoint["failure_errors"],
    )


def _validate_checkpoint_shapes(
    checkpoint: dict[str, np.ndarray],
    frame_indices: np.ndarray,
    success_count: int,
    failure_count: int,
) -> None:
    """Validate checkpoint array counts and shapes."""
    result_types = checkpoint["result_types"]
    failure_errors = checkpoint["failure_errors"]
    origins = checkpoint["origins"]
    analysis_radii = checkpoint["analysis_radii_pixels"]
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
    expected_indices = np.arange(result_types.shape[0], dtype=np.int64)
    if (
        frame_indices.shape != result_types.shape
        or not np.array_equal(frame_indices, expected_indices)
    ):
        raise ValueError(
            "VesEdge checkpoint frame indices are inconsistent."
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


def _successful_detections_from_checkpoint(
    origins: np.ndarray,
    analysis_radii: np.ndarray,
    full_values: np.ndarray,
    full_offsets: np.ndarray,
    frame_indices: np.ndarray,
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
                frame_index=int(frame_indices[index]),
            )
        )
    return detections


def _merge_checkpoint_results(
    result_types: np.ndarray,
    frame_indices: np.ndarray,
    successful: list[EdgeDetection],
    failure_errors: np.ndarray,
) -> list[EdgeResult]:
    """Restore original frame ordering of successes and failures."""
    detections: list[EdgeResult] = []
    success_index = 0
    failure_index = 0
    for result_type, frame_index in zip(
        result_types,
        frame_indices,
        strict=True,
    ):
        if int(result_type) == _DETECTION_CODE:
            detections.append(successful[success_index])
            success_index += 1
        else:
            detections.append(
                EdgeDetectionFailure(
                    str(failure_errors[failure_index]),
                    frame_index=int(frame_index),
                )
            )
            failure_index += 1
    return detections
