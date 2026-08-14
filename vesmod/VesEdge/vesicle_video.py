#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 10:42:28 2025.

@author: js2746
"""
from dataclasses import dataclass, field
from pathlib import Path
import traceback
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .vesicle_video_utils import convert_to_cartesian, measure_wrapped_finite_second_difference, downsample_to_new_indices
from .edge_filtering import EdgeQC

type EdgeResult = EdgeDetection | EdgeDetectionFailure

@dataclass
class EdgeDetection:
    """
    Detected edge on one frame of a `VesicleVideo`.

    Attributes
    ----------
    center : tuple(float, float)
        Cartesian coordinates (x, y) of the approximate vesicle center for each
        frame. Needed for wrapping images to/from polar coordinates.
    sampled_radii : NDArray[np.float64]
        The distances from `center` to the edge of the vesicle.
        Evenly spaced in theta, ranging from 0 to 2pi (not inclusive).
    sampled_theta : NDArray[np.float64]
        Evenly spaced angular values ranging from 0 to 2pi (not inclusive).
        Step size is determined by how many values are in `sampled_radii`.
    contour_x, contour_y : NDArray[np.float64]
        The Cartesian coordinates of the full detected edge prior to down-
        sampling. First and last bins overlap for plotting purposes.
    sampled_x, sampled_y : NDArray[np.float64]
        The Cartesian coordinates of the downsampled detected edge. First and
        last bins overlap for plotting purposes.
    qc : EdgeQC
        Quality control information from the `edge_filtering` module.
    median_radius : float
        The median radius value from this `EdgeDetection`.
    accepted : bool
        Whether or not this `EdgeDetection` passed QC in `edge_filtering`.
    """

    center: tuple[float, float]
    sampled_radii: NDArray[np.float64]
    contour_x: NDArray[np.float64]
    contour_y: NDArray[np.float64]
    qc: EdgeQC = field(default_factory=EdgeQC)

    @property
    def sampled_theta(self) -> NDArray[np.float64]:
        """Evenly spaced angular values ranging from 0 to 2pi (not inclusive)."""
        return np.linspace(0, 2 * np.pi, self.sampled_radii.shape[0], endpoint=False)

    @property
    def sampled_x(self) -> NDArray[np.float64]:
        """
        The x-coordinates of the contour represented as r(theta) on the \
        standardized angular grid.

        First and last bins overlap for plotting purposes.
        """
        x_vals = self.sampled_radii * np.cos(self.sampled_theta) + self.center[0]
        return np.append(x_vals, x_vals[0])

    @property
    def sampled_y(self) -> NDArray[np.float64]:
        """
        The y-coordinates of the contour represented as r(theta) on the \
        standardized angular grid.

        First and last bins overlap for plotting purposes.
        """
        y_vals = self.sampled_radii * np.sin(self.sampled_theta) + self.center[1]
        return np.append(y_vals, y_vals[0])

    @property
    def median_radius(self) -> float:
        """The median radius value from this `EdgeDetection`."""
        return float(np.median(self.sampled_radii))

    @property
    def accepted(self) -> bool:
        """Whether or not this `EdgeDetection` passed QC in `edge_filtering`."""
        return self.qc.passed


@dataclass(frozen=True)
class EdgeDetectionFailure:
    error: str


@dataclass
class VesicleVideo:
    """
    A class for vesicle videos.

    Holds both the raw images of a vesicle video, as well as its computed edges.

    Attributes
    ----------
        frames : numpy ndarray
            The 3D array of raw images. 0th dimension is frame number.
        micron_to_pixel_ratio : float
            The number of microns to pixels in your microscope image.
    """

    frames: np.ndarray
    micron_to_pixel_ratio: float
    n_angular_samples: int | None = 120
    detections: list[EdgeResult] = []

    def __post_init__(self):
        """
        Do argument validation.

        Raises
        ------
        TypeError
            If frames is not an ndarray.
        IndexError
            If frames is not a 3D ndarray.
            If n_angular_samples is greater than the number of samples.
        ValueError
            If pixel_to_micron_ratio is not positive.
            If n_angular_samples is not positive.
            if n_angular_samples is not an int or None.

        Returns
        -------
        None.

        """
        if not isinstance(self.frames, np.ndarray):
            raise TypeError("frames must be a numpy ndarray.")
        if len(self.frames.shape) != 3:
            raise IndexError("frames must be a 3D array.")
        if self.micron_to_pixel_ratio <= 0:
            raise ValueError("pixel_to_micron_ratio must be positive.")
        if isinstance(self.n_angular_samples, int):
            if self.n_angular_samples <= 0:
                raise ValueError("n_angular_samples must be positive")
            if self.n_angular_samples > self.frames.shape[1]:
                raise IndexError(f"Cannot downsample r_vals with len {self.frames.shape[1]} to {self.n_angular_samples}.")
        elif not isinstance(self.n_angular_samples, type(None)):
            raise ValueError("n_angular_samples must be an int or None.")

    def extract_edges(self, extractor_func, curvature_threshold=5):
        """
        Extract edges from every frame.

        Frames that fail edge extraction are marked with `EdgeDetectionFailure`.
        """
        for frame_num, frame in enumerate(self.frames):
            try:
                r_vals, vesicle_center = extractor_func(self.frames[frame_num, :, :])
                if r_vals.ndim != 1:
                    raise ValueError("Extractor must return a 1D array of r-values.")
                detected_edge = self._add_edge_to_video_frame(
                    frame_num,
                    r_vals,
                    vesicle_center,
                    curvature_threshold,
                )
            except Exception as error:
                print(f"Error on frame {frame_num}")
                traceback.print_exc()
                self.detections.append(EdgeDetectionFailure(str(error)))
                continue

            self._run_frame_qc(frame, detected_edge, curvature_threshold)
            self.detections.append(detected_edge)
        self._run_trajectory_qc()

    def _add_edge_to_video_frame(self, frame_num, r_vals, vesicle_center, curvature_threshold):
        """
        Save detected edge information for a given frame.

        Parameters
        ----------
        frame_num : int
            The frame number.
        r_vals : list or numpy ndarray
            The list or 1D array of radial distances from the vesicle_center,
            spaced evenly from 0 to 2pi.
        vesicle_center : tuple
            The origin (in y, x) of the polar coordinate system.
        curvature_threshold : float
            The level of curvature allowed between two contiguous r_vals before
            edge extraction would be deemed unreliable.

        Returns
        -------
        None.

        """
        x_vals, y_vals = convert_to_cartesian((vesicle_center[1], vesicle_center[0],), r_vals)
        if self.n_angular_samples is not None:
            r_vals = self._downsample_r_vals(r_vals, self.n_angular_samples)
        rescaled_r = r_vals * self.micron_to_pixel_ratio

        edge = EdgeDetection(
            (vesicle_center[1], vesicle_center[0]),
            rescaled_r,
            x_vals,
            y_vals,
        )
        
        self.detections[frame_num] = edge

        finite_second_difference = measure_wrapped_finite_second_difference(r_vals)
        if (np.fabs(finite_second_difference) >= curvature_threshold).any():
            self.status[frame_num] = 3
        elif np.isnan(finite_second_difference).any():
            self.status[frame_num] = 2
        else:
            self.status[frame_num] = 1

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
        if (show_trace and np.isnan(self.x_vals[0]).any()):
            raise ValueError("trace was set to True, but there are no edges detected for this vesicle.")
        output_path = path.with_suffix('.gif')
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_title(f"frame {i} / {self.frames.shape[0]}")
            ax.imshow(self.frames[i], cmap='gray', animated='True')
            if show_trace:
                if self.status[i] == 1:
                    ax.plot(self.x_vals[i], self.y_vals[i], color='tab:green')
                elif self.status[i] == 3:
                    ax.plot(self.x_vals[i], self.y_vals[i], color='tab:red')

        ani = FuncAnimation(fig, animate, frames=self.frames.shape[0], interval=150, blit=False, repeat_delay=1000)
        ani.save(output_path)
        plt.close()

    def save_edge_to_npy(self, path):
        """Save r_vals to a .npy file, removing frames with bad edge extraction."""
        if np.isnan(self.r_vals).all():
            raise AttributeError("Edge detection has not occurred, or went wrong.")
        output_values = []
        for frame_num, status in enumerate(self.status):
            if status == 1:
                output_values.append(self.r_vals[frame_num, :])
        np.save(path.with_suffix('.npy'), np.array(output_values))
