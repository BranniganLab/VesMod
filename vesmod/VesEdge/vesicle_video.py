#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 10:42:28 2025.

@author: js2746
"""
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
import traceback
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .vesicle_video_utils import downsample_to_new_indices
from .models import (
    EdgeDetection,
    EdgeDetectionFailure,
    EdgeResult,
    ImageContour,
)
from .edge_filtering import (
    EdgePopulationResult,
    EdgeQCConfig,
    check_curvature,
    check_edge_populations,
)


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
    A class for vesicle videos.

    Holds both the raw images of a vesicle video, as well as its computed edges.

    Attributes
    ----------
        frames : numpy ndarray
            The 3D array of raw images. 0th dimension is frame number.
        extraction_config : EdgeExtractionConfig
            The configuration parameters for the edge extractor.
        qc_config : EdgeQCConfig
            The configuration parameters for the quality control checks.
        detections : list[EdgeResult]
            The edge detections for each frame.
        population_result : EdgePopulationResult | None
            The relevant details from trajectory QC.
    """

    frames: np.ndarray
    extraction_config: EdgeExtractionConfig
    qc_config: EdgeQCConfig
    detections: list[EdgeResult] = field(default_factory=list)
    population_result: EdgePopulationResult | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        """
        Do argument validation.

        Raises
        ------
        TypeError
            If frames is not an ndarray.
        IndexError
            If frames is not a 3D ndarray.
            If n_angular_samples is greater than the number of samples.
        """
        if not isinstance(self.frames, np.ndarray):
            raise TypeError("frames must be a numpy ndarray.")
        if len(self.frames.shape) != 3:
            raise IndexError("frames must be a 3D array.")
        downsample_to = self.extraction_config.n_angular_samples
        if downsample_to is not None and downsample_to > self.frames.shape[1]:
            raise IndexError(f"Cannot downsample r_vals with len {self.frames.shape[1]} to {downsample_to}.")

    def extract_edges(
        self,
        extractor_func: Callable[
            [NDArray[np.float64]],
            tuple[NDArray[np.float64], tuple[float, float]],
        ],
    ) -> None:
        """
        Extract edges from every frame and save as `EdgeDetection`.

        Frames that produce errors are saved as `EdgeDetectionFailure`.

        Parameters
        ----------
        extractor_func : Callable
            The extractor function you wish to use to extract edges. Must take
            a 2D numpy array as an input (one frame). Must output a 1D NDArray
            of radii and a tuple containing the vesicle center in (y,x) format.

        Raises
        ------
        ValueError
            If no frame produced a successful detection.
            If successful detections have inconsistent angular sample counts.
            If no detection passed quality control.
        """
        self.detections = []
        for frame in self.frames:
            try:
                r_vals, vesicle_center = extractor_func(frame)
                self._validate_extractor_results(r_vals)
                detected_edge = self._compile_edge_detection_results(r_vals, vesicle_center)
            except Exception as error:
                traceback.print_exc()
                self.detections.append(EdgeDetectionFailure(str(error)))
                continue

            self._run_frame_qc(frame, detected_edge)
            self.detections.append(detected_edge)
        self._validate_detection_lengths()
        self._run_trajectory_qc()
        self._validate_usable_detections()

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
            The list or 1D array of radial distances from the vesicle_center,
            spaced evenly from 0 to 2pi.
        vesicle_center : tuple
            The origin (in y, x) of the polar coordinate system.

        Returns
        -------
        EdgeDetection

        """
        center = (vesicle_center[1], vesicle_center[0])
        full_contour = ImageContour(center, r_vals)
        if self.extraction_config.n_angular_samples is not None:
            downsampled_r_vals = self._downsample_r_vals(r_vals, self.extraction_config.n_angular_samples)
            analysis_contour = ImageContour(center, downsampled_r_vals)
            rescaled_r = downsampled_r_vals / self.extraction_config.pixels_per_micron
        else:
            analysis_contour = full_contour
            rescaled_r = r_vals / self.extraction_config.pixels_per_micron

        edge = EdgeDetection(full_contour, analysis_contour, rescaled_r)
        return edge

    def _downsample_r_vals(
        self,
        r_vals: NDArray[np.float64],
        n_samples: int = 120
    ) -> NDArray[np.float64]:
        """
        Downsample a vesicle edge profile to a fixed number of angular samples.

        The input edge profile is assumed to represent a periodic contour sampled
        at uniformly spaced angular positions. If the requested number of samples
        is smaller than the input length, the contour is resampled onto a new set
        of evenly spaced indices using linear interpolation with periodic wrapping.

        Parameters
        ----------
        r_vals : ndarray
            One-dimensional array of radial distances defining the vesicle edge.
            Each element corresponds to a uniformly spaced angular position.
        n_samples : int, optional
            Number of samples in the downsampled contour. Must be a positive
            integer less than or equal to ``len(r_vals)``. Default is 120.

        Returns
        -------
        ndarray
            One-dimensional array of length ``n_samples`` containing the
            downsampled radial distances.

        Notes
        -----
        If ``n_samples`` equals ``len(r_vals)``, the input array is returned
        unchanged. Periodic boundary conditions are assumed during
        interpolation so that the first and last angular samples are treated
        as adjacent points on a closed contour.

        """
        if n_samples == r_vals.shape[0]:
            return r_vals

        zero_to_ntheta = np.linspace(0, n_samples - 1, n_samples)
        new_evenly_spaced_indices = zero_to_ntheta * (r_vals.shape[0] / n_samples)
        downsampled_r_vals = downsample_to_new_indices(r_vals, new_evenly_spaced_indices)
        return downsampled_r_vals

    def _validate_extractor_results(self, r_vals: NDArray[np.float64]) -> None:
        """Check to make sure extractor returns a 1D ndarray."""
        if not isinstance(r_vals, np.ndarray):
            raise TypeError(f"Extractor must return an NDArray, not {type(r_vals)}.")
        if r_vals.ndim != 1:
            raise ValueError("Extractor must return a 1D array of r-values.")

    def _validate_detection_lengths(self) -> None:
        """Verify successful detections have consistent analysis-contour lengths."""
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

    def _run_frame_qc(self, frame: np.ndarray, edge: EdgeDetection) -> None:
        """
        Run quality-control checks that operate on a single detected edge.

        Applies all QC checks that require only the current video frame and its
        corresponding edge detection. Results are recorded in the detection's
        QC information.

        Parameters
        ----------
        frame : numpy.ndarray
            Image frame from which the edge was extracted.
        edge : EdgeDetection
            Edge detection to evaluate.

        Returns
        -------
        None
        """
        if self.qc_config.enable_curvature_qc:
            check_curvature(
                edge,
                threshold=self.qc_config.curvature_threshold,
            )

    def _run_trajectory_qc(self) -> None:
        """
        Run quality-control checks that require detections across video frames.

        Applies QC checks that use information from multiple edge detections in
        the video, such as identifying anomalous populations based on vesicle
        center and radius. Results are recorded in the QC information associated
        with the affected detections.

        Returns
        -------
        None
        """
        self.population_result = None

        if self.qc_config.enable_population_qc:
            self.population_result = check_edge_populations(
                self.detections,
                bic_threshold=self.qc_config.population_bic_threshold,
                max_minor_fraction=self.qc_config.max_minor_population_fraction,
            )

    def make_vesicle_gif(self, path: Path, show_trace: bool = True) -> None:
        """
        Make a .gif of the vesicle, with or without the detected edges shown.

        Parameters
        ----------
        path : pathlib Path
            The location and filename to save this .gif to.
        show_trace : Bool, optional
            Whether or not to display the detected edges. The default is True.

        Raises
        ------
        ValueError
            If ``show_trace`` is True and there is not exactly one detection
            result for every video frame.

        Returns
        -------
        None.

        """
        if not isinstance(path, Path):
            path = Path(path).resolve()
        output_path = path.with_suffix('.gif')

        if show_trace:
            if len(self.detections) != self.frames.shape[0]:
                raise ValueError(f"There are {len(self.detections)} detections and {self.frames.shape[0]} frames.")

        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_title(f"frame {i} / {self.frames.shape[0]}")
            ax.imshow(self.frames[i], cmap='gray', animated='True')
            if show_trace and isinstance(self.detections[i], EdgeDetection):
                contour = self.detections[i].full_contour
                if self.detections[i].accepted:
                    ax.plot(contour.x, contour.y, color='tab:green')
                else:
                    ax.plot(contour.x, contour.y, color='tab:red')

        ani = FuncAnimation(fig, animate, frames=self.frames.shape[0], interval=150, blit=False, repeat_delay=1000)
        ani.save(output_path)
        plt.close()

    def save_edge_to_npy(self, path: Path) -> None:
        """
        Save accepted radii to a .npy file.

        Frames with failed edge extraction and frames rejected by quality
        control are excluded.

        Raises
        ------
        ValueError
            If no accepted edge detection is available.
        """
        output_values = []
        if not isinstance(path, Path):
            path = Path(path).resolve()
        for edge in self.detections:
            if isinstance(edge, EdgeDetection) and edge.accepted:
                output_values.append(edge.radii_microns)
        if not output_values:
            raise ValueError(
                "Cannot save edges: no accepted edge detections are available."
            )
        np.save(path.with_suffix('.npy'), np.array(output_values))
