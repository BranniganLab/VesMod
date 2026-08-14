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
from .edge_filtering import EdgeQC, EdgeQCConfig

type EdgeResult = EdgeDetection | EdgeDetectionFailure


@dataclass(frozen=True)
class ImageContour:
    """
    The polar and Cartesian coordinates of a contour in the frame of reference \
    of the image they are extracted from.

    Attributes
    ----------
    origin : tuple(float, float)
        Cartesian coordinates (x, y) of the origin point in the original image's
        frame of reference.
    r : NDArray[np.float64]
        The distances from `center` to the edge of the vesicle. Evenly spaced
        in theta, ranging from 0 to 2pi (not inclusive).
    theta : NDArray[np.float64]
        Evenly spaced angular values ranging from 0 to 2pi (not inclusive).
        Step size is determined by how many values are in `r`.
    x, y : NDArray[np.float64]
        The Cartesian coordinates of the Contour. First and last bins overlap
        for plotting purposes.
    """

    origin: tuple[float, float]
    r: NDArray[np.float64]

    @property
    def theta(self) -> NDArray[np.float64]:
        """Evenly spaced angular values ranging from 0 to 2pi (not inclusive)."""
        return np.linspace(0, 2 * np.pi, self.r.shape[0], endpoint=False)

    @property
    def x(self) -> NDArray[np.float64]:
        """
        The x-coordinates of the contour.

        First and last bins overlap for plotting purposes.
        """
        x_vals = self.r * np.cos(self.theta) + self.origin[0]
        return np.append(x_vals, x_vals[0])

    @property
    def y(self) -> NDArray[np.float64]:
        """
        The y-coordinates of the contour.

        First and last bins overlap for plotting purposes.
        """
        y_vals = self.r * np.sin(self.theta) + self.origin[1]
        return np.append(y_vals, y_vals[0])


@dataclass
class EdgeDetection:
    """
    Detected edge on one frame of a `VesicleVideo`.

    Attributes
    ----------
    contour : VesicleContour
        The full detected contour.
    analysis_contour : VesicleContour
        The contour points after downsampling (if performed).
    radii_microns : NDArray[np.float64]
        The radial profile of the vesicle, derived from the `analysis_contour`
        and the `pixels_per_micron`.
    qc : EdgeQC
        Quality control information from the `edge_filtering` module.
    median_radius : float
        The median radius value from this `EdgeDetection`.
    accepted : bool
        Whether or not this `EdgeDetection` passed QC in `edge_filtering`.
    """

    full_contour: ImageContour
    analysis_contour: ImageContour
    radii_microns: NDArray[np.float64]
    qc: EdgeQC = field(default_factory=EdgeQC)

    @property
    def median_radius(self) -> float:
        """The median radius value from this `EdgeDetection`."""
        return float(np.median(self.radii_microns))

    @property
    def accepted(self) -> bool:
        """Whether or not this `EdgeDetection` passed QC in `edge_filtering`."""
        return self.qc.passed


@dataclass(frozen=True)
class EdgeDetectionFailure:
    """Holds an error message upon Edge Detection failing due to an error."""

    error: str


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
        """
        Do argument validation.

        Raises
        ------
        TypeError
            If `n_angular_samples` is not `int` or `None`.
        ValueError
            If `pixels_per_micron` is not positive.
            If `n_angular_samples` is not positive.
        """
        if self.pixels_per_micron <= 0:
            raise ValueError("pixels_per_micron must be positive.")
        if isinstance(self.n_angular_samples, int):
            if self.n_angular_samples <= 0:
                raise ValueError("n_angular_samples must be positive")
        elif not isinstance(self.n_angular_samples, type(None)):
            raise TypeError("n_angular_samples must be an int or None.")


@dataclass
class VesicleVideo:
    """
    A class for vesicle videos.

    Holds both the raw images of a vesicle video, as well as its computed edges.

    Attributes
    ----------
        frames : numpy ndarray
            The 3D array of raw images. 0th dimension is frame number.
        extraction_config : EdgeExtractorConfig
            The configuration parameters for the edge extractor.
        qc_config : EdgeQCConfig
            The configuration parameters for the quality control checks.
        detections : list[EdgeResult]
            The edge detections for each frame.
    """

    frames: np.ndarray
    extraction_config = EdgeExtractionConfig
    qc_config: EdgeQCConfig
    detections: list[EdgeResult] = []

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
        if self.extraction_config.n_angular_samples > self.frames.shape[1]:
            raise IndexError(f"Cannot downsample r_vals with len {self.frames.shape[1]} to {self.extraction_config.n_angular_samples}.")

    def extract_edges(self, extractor_func: Callable[NDArray[np.float64]]):
        """
        Extract edges from every frame and save as `EdgeDetection`.

        Frames that produce errors are saved as `EdgeDetectionFailure`.

        Parameters
        ----------
        extractor_func : Callable
            The extractor function you wish to use to extract edges. Must take
            a 2D numpy array as an input (one frame). Must output a 1D NDArray
            of radii and a tuple containing the vesicle center in (y,x) format.
        """
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
        self._run_trajectory_qc()

    def _compile_edge_detection_results(self, r_vals, vesicle_center):
        """
        Save detected edge information for a given frame.

        Parameters
        ----------
        r_vals : list or numpy ndarray
            The list or 1D array of radial distances from the vesicle_center,
            spaced evenly from 0 to 2pi.
        vesicle_center : tuple
            The origin (in y, x) of the polar coordinate system.

        Returns
        -------
        None.

        """
        center = (vesicle_center[1], vesicle_center[0])
        full_contour = ImageContour(center, r_vals)
        if self.extraction_config.n_angular_samples is not None:
            downsampled_r_vals = self._downsample_r_vals(r_vals, self.extraction_config.n_angular_samples)
            analysis_contour = ImageContour(center, downsampled_r_vals)
            rescaled_r = downsampled_r_vals / self.extraction_config.pixels_per_micron
        else:
            analysis_contour = full_contour
            rescaled_r = r_vals * self.extraction_config.pixels_per_micron

        edge = EdgeDetection(full_contour, analysis_contour, rescaled_r)
        return edge

    def _downsample_r_vals(self, r_vals: np.ndarray, n_samples: int = 120) -> np.ndarray:
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

    def _validate_extractor_results(self, r_vals):
        """Check to make sure extractor returns a 1D ndarray."""
        if not isinstance(r_vals, np.ndarray):
            raise TypeError(f"Extractor must return an NDArray, not {type(r_vals)}.")
        if r_vals.ndim != 1:
            raise ValueError("Extractor must return a 1D array of r-values.")

    def make_vesicle_gif(self, path, show_trace=True):
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
            If show_trace is True, but there are no edges saved.

        Returns
        -------
        None.

        """
        if not isinstance(path, Path):
            path = Path(path).resolve()
        output_path = path.with_suffix('.gif')
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_title(f"frame {i} / {self.frames.shape[0]}")
            ax.imshow(self.frames[i], cmap='gray', animated='True')
            if show_trace:
                contour = self.detections[i].full_contour
                if self.detections[i].accepted:
                    ax.plot(contour.x, contour.y, color='tab:green')
                else:
                    ax.plot(contour.x, contour.y, color='tab:red')

        ani = FuncAnimation(fig, animate, frames=self.frames.shape[0], interval=150, blit=False, repeat_delay=1000)
        ani.save(output_path)
        plt.close()

    def save_edge_to_npy(self, path):
        """Save radii to a .npy file, removing frames with bad edge extraction."""
        output_values = []
        for edge in self.detections:
            if edge.accepted:
                output_values.append(edge.radii_microns)
        np.save(path.with_suffix('.npy'), np.array(output_values))
