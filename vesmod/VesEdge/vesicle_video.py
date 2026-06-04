#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 10:42:28 2025.

@author: js2746
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .vesicle_video_utils import convert_to_cartesian, measure_wrapped_finite_second_difference


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
        vesicle_centers : list of tuples
            List of len(frames.shape[0]) containing Cartesian coordinates of the
            approximate vesicle center for each frame. Needed for wrapping images
            to/from polar coordinates.
        r_vals : numpy ndarray
            The distance from vesicle_center on frame i to the edge of the vesicle
            on frame i. Evenly spaced in theta, ranging from 0 to 2pi.
        x_vals, y_vals : numpy ndarrays
            The Cartesian coordinates of the vesicle edge.
        status : list[int]
            List of ints containing status code for each frame. 1 = useable frame,
            2 = error on edge extraction, 3 = unreliable edge extraction.
    """

    frames: np.ndarray
    micron_to_pixel_ratio: float
    vesicle_centers: list = field(init=False)
    r_vals: np.ndarray = field(init=False)
    x_vals: np.ndarray = field(init=False)
    y_vals: np.ndarray = field(init=False)
    status: list = field(init=False)

    def __post_init__(self):
        """
        Do argument validation on frames. Initialize all else to None, nan, or 0.

        Raises
        ------
        TypeError
            If frames is not an ndarray.
        IndexError
            If frames is not a 3D ndarray.

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
        self.vesicle_centers = [None] * self.frames.shape[0]
        self.r_vals = np.full((self.frames.shape[0], self.frames.shape[1]), np.nan)
        self.x_vals = np.full((self.frames.shape[0], self.frames.shape[1]), np.nan)
        self.y_vals = np.full((self.frames.shape[0], self.frames.shape[1]), np.nan)
        self.status = np.zeros(self.frames.shape[0]).astype(int)

    def extract_edges(self, extractor_func, curvature_threshold=10):
        """
        Extract edges from every frame.

        Parameters
        ----------
        extractor_func : function
            The edge extractor function you wish to use.
        curvature_threshold : float, OPTIONAL
            The level of curvature allowed between two contiguous r_vals before
            edge extraction would be deemed unreliable. Default is 10.

        Returns
        -------
        None.

        Side Effects
        ------------
        - Saves self.r_vals with units of microns
        - If error encountered on a frame, sets self.status to 2 for that frame.

        """
        for frame_num, _ in enumerate(self.frames):
            try:
                r_vals, vesicle_center = extractor_func(self.frames[frame_num, :, :])
                self._add_edge_to_video_frame(frame_num, r_vals, vesicle_center, curvature_threshold)
            except ValueError:
                print(f"Error on frame {frame_num}")
                self.status[frame_num] = 2

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
            The origin (in x, y) of the polar coordinate system.
        curvature_threshold : float
            The level of curvature allowed between two contiguous r_vals before
            edge extraction would be deemed unreliable.

        Returns
        -------
        None.

        """
        self.r_vals[frame_num] = r_vals * self.micron_to_pixel_ratio
        self.vesicle_centers[frame_num] = vesicle_center
        self.x_vals[frame_num], self.y_vals[frame_num] = convert_to_cartesian((vesicle_center[1], vesicle_center[0],), r_vals)
        if (measure_wrapped_finite_second_difference(r_vals) > curvature_threshold).any():
            self.status[frame_num] = 3
        else:
            self.status[frame_num] = 1

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
