#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 10:42:28 2025.

@author: js2746
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import traceback

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.typing import NDArray

from .edge_filtering import (
    EdgePopulationResult,
    EdgeQCConfig,
    check_curvature,
    check_edge_populations,
)
from .models import (
    EdgeDetection,
    EdgeDetectionFailure,
    EdgeQC,
    EdgeResult,
    ImageContour,
)
from .vesicle_video_utils import downsample_to_new_indices

CHECKPOINT_VERSION = 1
_DETECTION_CODE = 1
_FAILURE_CODE = 0


@dataclass(frozen=True)
class EdgeExtractionConfig:
    """
    Configuration parameters for the edge extractor.

    Attributes
    ----------
    pixels_per_micron : float
        How many pixels in the image represent one micron in real space.
    n_angular_samples : int | None
        How many angular samples to downsample to. If None, do not downsample.
    """

    pixels_per_micron: float = 1
    n_angular_samples: int | None = 120

    def __post_init__(self) -> None:
        """Validate and normalize edge-extraction configuration."""
        if not np.isfinite(self.pixels_per_micron):
            raise ValueError("pixels_per_micron must be finite.")
        if self.pixels_per_micron <= 0:
            raise ValueError("pixels_per_micron must be positive.")

        if self.n_angular_samples is None:
            return

        if not isinstance(
            self.n_angular_samples,
            (int, float),
        ):
            raise TypeError(
                "n_angular_samples must be an integer-valued number or None."
            )

        if not np.isfinite(self.n_angular_samples):
            raise ValueError(
                "n_angular_samples must be finite."
            )

        if not float(self.n_angular_samples).is_integer():
            raise ValueError(
                "n_angular_samples must be integer-valued."
            )

        n_angular_samples = int(
            self.n_angular_samples
        )

        if n_angular_samples <= 0:
            raise ValueError(
                "n_angular_samples must be positive."
            )

        object.__setattr__(
            self,
            "n_angular_samples",
            n_angular_samples,
        )


@dataclass
class VesicleVideo:
    """
    A class for vesicle videos and previously extracted vesicle edges.

    A video created from image data stores its raw frames as well as its edge
    detections. A video restored from a checkpoint has ``frames=None`` but
    retains the extracted edge information required to rerun quality control.

    Attributes
    ----------
    frames : numpy.ndarray | None
        Three-dimensional array of raw images. Axis 0 is frame number. None for
        videos restored from a checkpoint without image data.
    extraction_config : EdgeExtractionConfig
        Configuration parameters used to construct analysis contours.
    qc_config : EdgeQCConfig
        Configuration parameters for quality-control checks.
    detections : list[EdgeResult]
        Edge-extraction result corresponding to each original video frame.
    population_result : EdgePopulationResult | None
        Details from trajectory-level population QC, if it was run.
    """

    frames: np.ndarray | None
    extraction_config: EdgeExtractionConfig
    qc_config: EdgeQCConfig
    detections: list[EdgeResult] = field(default_factory=list)
    population_result: EdgePopulationResult | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        """
        Validate image frames when they are available.

        Raises
        ------
        TypeError
            If ``frames`` is neither a NumPy array nor None.
        IndexError
            If ``frames`` is not three-dimensional, or if the configured
            analysis sample count exceeds the image dimension used by the
            existing constructor validation.
        """
        if self.frames is None:
            return
        if not isinstance(self.frames, np.ndarray):
            raise TypeError("frames must be a numpy ndarray or None.")
        if self.frames.ndim != 3:
            raise IndexError("frames must be a 3D array.")

        downsample_to = self.extraction_config.n_angular_samples
        if (
            downsample_to is not None
            and downsample_to > self.frames.shape[1]
        ):
            raise IndexError(
                "Cannot downsample r_vals with len "
                f"{self.frames.shape[1]} to {downsample_to}."
            )

    def extract_edges(
        self,
        extractor_func: Callable[
            [NDArray[np.float64]],
            tuple[NDArray[np.float64], tuple[float, float]],
        ],
    ) -> None:
        """
        Extract edges from every image frame and run quality control.

        Frames that produce errors are saved as `EdgeDetectionFailure`.

        Parameters
        ----------
        extractor_func : Callable
            Edge extractor that accepts one two-dimensional image frame and
            returns one-dimensional radii plus the vesicle center in (y, x)
            order.

        Raises
        ------
        ValueError
            If image frames are unavailable, no frame produced a successful
            detection, successful detections have inconsistent analysis-contour
            lengths, or no detection passed quality control.
        """
        if self.frames is None:
            raise ValueError(
                "Cannot extract edges: image frames are not available."
            )

        self.detections = []
        for frame in self.frames:
            try:
                r_vals, vesicle_center = extractor_func(frame)
                self._validate_extractor_results(r_vals)
                detected_edge = self._compile_edge_detection_results(
                    r_vals,
                    vesicle_center,
                )
            except Exception as error:
                traceback.print_exc()
                self.detections.append(
                    EdgeDetectionFailure(str(error))
                )
                continue

            self.detections.append(detected_edge)

        self.run_qc()

    def run_qc(
        self,
        qc_config: EdgeQCConfig | None = None,
    ) -> None:
        """
        Run enabled quality-control checks on existing edge detections.

        Existing QC results are cleared before checks are rerun.

        Parameters
        ----------
        qc_config : EdgeQCConfig | None, optional
            New QC configuration to use. If None, use the video's current
            configuration.

        Raises
        ------
        ValueError
            If there are no successful detections, if successful detections
            have inconsistent angular sample counts, or if no detection passes
            quality control.
        """
        if qc_config is not None:
            self.qc_config = qc_config

        self._validate_detection_lengths()
        self._reset_qc()

        for detection in self.detections:
            if isinstance(detection, EdgeDetection):
                self._run_frame_qc(detection)

        self._run_trajectory_qc()
        self._validate_usable_detections()

    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save unfiltered VesEdge state required to rerun quality control.

        The checkpoint is a versioned ``.npz`` archive. It preserves all
        successful detections, extraction failures, frame ordering, image-space
        centers, native contours, analysis contours, physical radii, and the
        extraction and QC configurations. Raw image frames and existing QC
        results are not saved.

        Parameters
        ----------
        path : str | pathlib.Path
            Output path. The suffix is replaced with ``.npz``.

        Raises
        ------
        ValueError
            If no successful detection exists or successful analysis contours
            have inconsistent lengths.
        """
        self._validate_detection_lengths()
        output_path = Path(path).with_suffix(".npz")
        successful = [
            result
            for result in self.detections
            if isinstance(result, EdgeDetection)
        ]
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
        full_radii_values, full_radii_offsets = (
            self._flatten_full_radii(successful)
        )
        n_angular_samples = (
            -1
            if self.extraction_config.n_angular_samples is None
            else self.extraction_config.n_angular_samples
        )

        np.savez(
            output_path,
            checkpoint_version=np.asarray(CHECKPOINT_VERSION),
            pixels_per_micron=np.asarray(
                self.extraction_config.pixels_per_micron,
                dtype=float,
            ),
            n_angular_samples=np.asarray(
                n_angular_samples,
                dtype=np.int64,
            ),
            curvature_threshold=np.asarray(
                self.qc_config.curvature_threshold,
                dtype=float,
            ),
            population_bic_threshold=np.asarray(
                self.qc_config.population_bic_threshold,
                dtype=float,
            ),
            max_minor_population_fraction=np.asarray(
                self.qc_config.max_minor_population_fraction,
                dtype=float,
            ),
            enable_curvature_qc=np.asarray(
                self.qc_config.enable_curvature_qc,
                dtype=bool,
            ),
            enable_population_qc=np.asarray(
                self.qc_config.enable_population_qc,
                dtype=bool,
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
    def from_checkpoint(
        cls,
        path: str | Path,
        qc_config: EdgeQCConfig | None = None,
    ) -> "VesicleVideo":
        """
        Restore extracted edges from a VesEdge checkpoint and run QC.

        Parameters
        ----------
        path : str | pathlib.Path
            Path to a checkpoint produced by :meth:`save_checkpoint`.
        qc_config : EdgeQCConfig | None, optional
            QC settings to apply after loading. If None, reuse the QC settings
            stored in the checkpoint.

        Returns
        -------
        VesicleVideo
            Video with ``frames=None`` and reconstructed per-frame extraction
            results. QC has been rerun using the selected configuration.

        Raises
        ------
        FileNotFoundError
            If the checkpoint does not exist.
        ValueError
            If the path is not an ``.npz`` file, the checkpoint version is not
            supported, checkpoint data are inconsistent, or no loaded detection
            passes the selected QC settings.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )
        if checkpoint_path.suffix != ".npz":
            raise ValueError("VesEdge checkpoints must end in .npz.")

        with np.load(
            checkpoint_path,
            allow_pickle=False,
        ) as checkpoint:
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

        extraction_config = cls._extraction_config_from_checkpoint(
            saved_data
        )
        stored_qc_config = cls._qc_config_from_checkpoint(
            saved_data
        )
        selected_qc_config = (
            stored_qc_config
            if qc_config is None
            else qc_config
        )

        video = cls(
            frames=None,
            extraction_config=extraction_config,
            qc_config=selected_qc_config,
        )
        video.detections = cls._detections_from_checkpoint(
            saved_data
        )
        video.run_qc()
        return video

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
        """Verify that a checkpoint contains all fields required to reload."""
        required_keys = {
            "checkpoint_version",
            "pixels_per_micron",
            "n_angular_samples",
            "curvature_threshold",
            "population_bic_threshold",
            "max_minor_population_fraction",
            "enable_curvature_qc",
            "enable_population_qc",
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
        """Reconstruct edge-extraction settings stored in a checkpoint."""
        stored_samples = int(
            checkpoint["n_angular_samples"]
        )
        n_angular_samples = (
            None
            if stored_samples == -1
            else stored_samples
        )
        return EdgeExtractionConfig(
            pixels_per_micron=float(
                checkpoint["pixels_per_micron"]
            ),
            n_angular_samples=n_angular_samples,
        )

    @staticmethod
    def _qc_config_from_checkpoint(
        checkpoint: dict[str, np.ndarray],
    ) -> EdgeQCConfig:
        """Reconstruct quality-control settings stored in a checkpoint."""
        return EdgeQCConfig(
            curvature_threshold=float(
                checkpoint["curvature_threshold"]
            ),
            population_bic_threshold=float(
                checkpoint["population_bic_threshold"]
            ),
            max_minor_population_fraction=float(
                checkpoint["max_minor_population_fraction"]
            ),
            enable_curvature_qc=bool(
                checkpoint["enable_curvature_qc"]
            ),
            enable_population_qc=bool(
                checkpoint["enable_population_qc"]
            ),
        )

    @classmethod
    def _detections_from_checkpoint(
        cls,
        checkpoint: dict[str, np.ndarray],
    ) -> list[EdgeResult]:
        """Reconstruct ordered extraction results stored in a checkpoint."""
        result_types = checkpoint["result_types"]
        failure_errors = checkpoint["failure_errors"]
        origins = checkpoint["origins"]
        analysis_radii = checkpoint["analysis_radii_pixels"]
        radii_microns = checkpoint["radii_microns"]
        full_values = checkpoint["full_radii_values"]
        full_offsets = checkpoint["full_radii_offsets"]

        success_count = int(
            np.count_nonzero(result_types == _DETECTION_CODE)
        )
        failure_count = int(
            np.count_nonzero(result_types == _FAILURE_CODE)
        )
        cls._validate_checkpoint_shapes(
            result_types,
            failure_errors,
            origins,
            analysis_radii,
            radii_microns,
            full_offsets,
            full_values,
            success_count,
            failure_count,
        )

        successful = cls._successful_detections_from_checkpoint(
            origins,
            analysis_radii,
            radii_microns,
            full_values,
            full_offsets,
        )
        return cls._merge_checkpoint_results(
            result_types,
            successful,
            failure_errors,
        )

    @staticmethod
    def _validate_checkpoint_shapes(
        result_types: np.ndarray,
        failure_errors: np.ndarray,
        origins: np.ndarray,
        analysis_radii: np.ndarray,
        radii_microns: np.ndarray,
        full_offsets: np.ndarray,
        full_values: np.ndarray,
        success_count: int,
        failure_count: int,
    ) -> None:
        """Validate array counts and shapes used to reconstruct detections."""
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
            full_contour = ImageContour(
                origin,
                full_values[start:stop].copy(),
            )
            analysis_contour = ImageContour(
                origin,
                analysis_radii[index].copy(),
            )
            detections.append(
                EdgeDetection(
                    full_contour,
                    analysis_contour,
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
        """Restore successful detections and failures to original order."""
        detections: list[EdgeResult] = []
        success_index = 0
        failure_index = 0
        for result_type in result_types:
            if result_type == _DETECTION_CODE:
                detections.append(
                    successful[success_index]
                )
                success_index += 1
            else:
                detections.append(
                    EdgeDetectionFailure(
                        str(failure_errors[failure_index])
                    )
                )
                failure_index += 1
        return detections

    def _reset_qc(self) -> None:
        """Clear all stored quality-control results."""
        self.population_result = None

        for detection in self.detections:
            if isinstance(detection, EdgeDetection):
                detection.qc = EdgeQC()

    def _compile_edge_detection_results(
        self,
        r_vals: NDArray[np.float64],
        vesicle_center: tuple[float, float],
    ) -> EdgeDetection:
        """
        Save detected edge information for a given frame.

        Parameters
        ----------
        r_vals : numpy ndarray
            One-dimensional radial distances from the vesicle center, spaced
            evenly from 0 to 2pi.
        vesicle_center : tuple
            Origin in (y, x) image-coordinate order.

        Returns
        -------
        EdgeDetection
            Successful edge detection with native and analysis contours.
        """
        center = (vesicle_center[1], vesicle_center[0])
        full_contour = ImageContour(center, r_vals)
        if self.extraction_config.n_angular_samples is not None:
            downsampled_r_vals = self._downsample_r_vals(
                r_vals,
                self.extraction_config.n_angular_samples,
            )
            analysis_contour = ImageContour(center, downsampled_r_vals)
            rescaled_r = (
                downsampled_r_vals
                / self.extraction_config.pixels_per_micron
            )
        else:
            analysis_contour = full_contour
            rescaled_r = (
                r_vals
                / self.extraction_config.pixels_per_micron
            )

        return EdgeDetection(
            full_contour,
            analysis_contour,
            rescaled_r,
        )

    def _downsample_r_vals(
        self,
        r_vals: NDArray[np.float64],
        n_samples: int = 120,
    ) -> NDArray[np.float64]:
        """
        Downsample a vesicle edge profile to a fixed number of angular samples.

        The input edge profile is assumed to represent a periodic contour
        sampled at uniformly spaced angular positions. If the requested number
        of samples is smaller than the input length, the contour is resampled
        onto evenly spaced indices using linear interpolation with periodic
        wrapping.
        """
        if n_samples == r_vals.shape[0]:
            return r_vals

        zero_to_ntheta = np.linspace(
            0,
            n_samples - 1,
            n_samples,
        )
        new_evenly_spaced_indices = (
            zero_to_ntheta
            * (r_vals.shape[0] / n_samples)
        )
        return downsample_to_new_indices(
            r_vals,
            new_evenly_spaced_indices,
        )

    def _validate_extractor_results(
        self,
        r_vals: NDArray[np.float64],
    ) -> None:
        """Check that an extractor returned a one-dimensional NumPy array."""
        if not isinstance(r_vals, np.ndarray):
            raise TypeError(
                "Extractor must return an NDArray, "
                f"not {type(r_vals)}."
            )
        if r_vals.ndim != 1:
            raise ValueError(
                "Extractor must return a 1D array of r-values."
            )

    def _validate_detection_lengths(self) -> None:
        """Verify successful detections have consistent analysis lengths."""
        unique_lengths = {
            detection.radii_microns.shape[0]
            for detection in self.detections
            if isinstance(detection, EdgeDetection)
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
        """Verify that at least one detected edge passes quality control."""
        if not any(
            isinstance(detection, EdgeDetection)
            and detection.accepted
            for detection in self.detections
        ):
            raise ValueError(
                "Edge extraction produced detections, but no frames passed "
                "quality control."
            )

    def _run_frame_qc(self, edge: EdgeDetection) -> None:
        """Run configured quality-control checks on one detected edge."""
        if self.qc_config.enable_curvature_qc:
            check_curvature(
                edge,
                threshold=self.qc_config.curvature_threshold,
            )

    def _run_trajectory_qc(self) -> None:
        """Run configured quality-control checks across video detections."""
        self.population_result = None

        if self.qc_config.enable_population_qc:
            self.population_result = check_edge_populations(
                self.detections,
                bic_threshold=(
                    self.qc_config.population_bic_threshold
                ),
                max_minor_fraction=(
                    self.qc_config.max_minor_population_fraction
                ),
            )

    def make_vesicle_gif(
        self,
        path: str | Path,
        show_trace: bool = True,
    ) -> None:
        """
        Make a GIF of the image frames, optionally with detected edges shown.

        Raises
        ------
        ValueError
            If image frames are unavailable, or if ``show_trace`` is True and
            there is not exactly one detection result per image frame.
        """
        if self.frames is None:
            raise ValueError(
                "Cannot make vesicle GIF: image frames are not available."
            )

        output_path = Path(path).with_suffix(".gif")
        if (
            show_trace
            and len(self.detections) != self.frames.shape[0]
        ):
            raise ValueError(
                f"There are {len(self.detections)} detections and "
                f"{self.frames.shape[0]} frames."
            )

        fig, ax = plt.subplots()

        def animate(index):
            ax.clear()
            ax.set_title(
                f"frame {index} / {self.frames.shape[0]}"
            )
            ax.imshow(
                self.frames[index],
                cmap="gray",
                animated="True",
            )
            if (
                show_trace
                and isinstance(
                    self.detections[index],
                    EdgeDetection,
                )
            ):
                contour = self.detections[index].full_contour
                if self.detections[index].accepted:
                    ax.plot(
                        contour.x,
                        contour.y,
                        color="tab:green",
                    )
                else:
                    ax.plot(
                        contour.x,
                        contour.y,
                        color="tab:red",
                    )

        ani = FuncAnimation(
            fig,
            animate,
            frames=self.frames.shape[0],
            interval=150,
            blit=False,
            repeat_delay=1000,
        )
        ani.save(output_path)
        plt.close()

    def save_edge_to_npy(self, path: str | Path) -> None:
        """
        Save accepted physical radii to a Spectrum-ready ``.npy`` file.

        Frames with failed extraction and frames rejected by quality control are
        excluded.

        Raises
        ------
        ValueError
            If no accepted edge detection is available.
        """
        output_values = [
            edge.radii_microns
            for edge in self.detections
            if isinstance(edge, EdgeDetection)
            and edge.accepted
        ]
        if not output_values:
            raise ValueError(
                "Cannot save edges: no accepted edge detections are available."
            )
        np.save(
            Path(path).with_suffix(".npy"),
            np.asarray(output_values),
        )
