#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw vesicle video data and edge-extraction orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import traceback

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.typing import NDArray

from .models import (
    EdgeDetection,
    EdgeDetectionFailure,
    EdgeResult,
    ImageContour,
)
from .vesicle_video_utils import downsample_to_new_indices


@dataclass(frozen=True)
class EdgeExtractionConfig:
    """Configuration parameters for edge extraction and contour preparation."""

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
        if not isinstance(self.n_angular_samples, (int, float)):
            raise TypeError(
                "n_angular_samples must be an integer-valued number or None."
            )
        if not np.isfinite(self.n_angular_samples):
            raise ValueError("n_angular_samples must be finite.")
        if not float(self.n_angular_samples).is_integer():
            raise ValueError("n_angular_samples must be integer-valued.")

        n_angular_samples = int(self.n_angular_samples)
        if n_angular_samples <= 0:
            raise ValueError("n_angular_samples must be positive.")
        object.__setattr__(self, "n_angular_samples", n_angular_samples)


@dataclass
class VesicleVideo:
    """Raw image frames for one vesicle trajectory."""

    frames: np.ndarray

    def __post_init__(self) -> None:
        """Validate raw image frames."""
        if not isinstance(self.frames, np.ndarray):
            raise TypeError("frames must be a numpy ndarray.")
        if self.frames.ndim != 3:
            raise IndexError("frames must be a 3D array.")

    def extract_edges(
        self,
        extractor_func: Callable[
            [NDArray[np.float64]],
            tuple[NDArray[np.float64], tuple[float, float]],
        ],
        extraction_config: EdgeExtractionConfig,
    ) -> "VesicleEdges":
        """Extract an edge from each frame without running quality control.

        Parameters
        ----------
        extractor_func : Callable
            Function accepting one 2D image frame and returning one-dimensional
            radii plus the vesicle center in ``(y, x)`` order.
        extraction_config : EdgeExtractionConfig
            Configuration used to downsample and convert extracted radii.

        Returns
        -------
        VesicleEdges
            Ordered extraction results ready for checkpointing or QC.

        Raises
        ------
        ValueError
            If no frame produces a successful detection or successful analysis
            contours have inconsistent lengths.
        """
        from .vesicle_edges import VesicleEdges

        detections: list[EdgeResult] = []
        for frame in self.frames:
            try:
                r_vals, vesicle_center = extractor_func(frame)
                self._validate_extractor_results(r_vals)
                detected_edge = self._compile_edge_detection_results(
                    r_vals,
                    vesicle_center,
                    extraction_config,
                )
            except Exception as error:
                traceback.print_exc()
                detections.append(EdgeDetectionFailure(str(error)))
                continue
            detections.append(detected_edge)

        return VesicleEdges(
            extraction_config=extraction_config,
            detections=detections,
        )

    def make_vesicle_gif(
        self,
        path: str | Path,
        edges: "VesicleEdges | None" = None,
    ) -> None:
        """Save a GIF of the video, optionally overlaying extracted edges.

        Parameters
        ----------
        path : str | pathlib.Path
            Output path. The suffix is replaced with ``.gif``.
        edges : VesicleEdges | None, optional
            Extracted edges to overlay. Successful detections are green when
            accepted by a QC run and red when rejected. Before QC, successful
            detections are shown in green.

        Raises
        ------
        ValueError
            If supplied extraction results do not match the frame count.
        """
        if edges is not None and len(edges.detections) != self.frames.shape[0]:
            raise ValueError(
                f"There are {len(edges.detections)} detections and "
                f"{self.frames.shape[0]} frames."
            )

        output_path = Path(path).with_suffix(".gif")
        fig, ax = plt.subplots()

        def animate(index: int) -> None:
            ax.clear()
            ax.set_title(f"frame {index} / {self.frames.shape[0]}")
            ax.imshow(self.frames[index], cmap="gray", animated=True)
            if edges is None:
                return
            result = edges.detections[index]
            if not isinstance(result, EdgeDetection):
                return
            contour = result.full_contour
            color = "tab:green"
            if edges.qc_config is not None and not result.accepted:
                color = "tab:red"
            ax.plot(contour.x, contour.y, color=color)

        animation = FuncAnimation(
            fig,
            animate,
            frames=self.frames.shape[0],
            interval=150,
            blit=False,
            repeat_delay=1000,
        )
        animation.save(output_path)
        plt.close()

    @staticmethod
    def _validate_extractor_results(
        r_vals: NDArray[np.float64],
    ) -> None:
        """Verify that an extractor returned a one-dimensional ndarray."""
        if not isinstance(r_vals, np.ndarray):
            raise TypeError(
                f"Extractor must return an NDArray, not {type(r_vals)}."
            )
        if r_vals.ndim != 1:
            raise ValueError("Extractor must return a 1D array of r-values.")

    @classmethod
    def _compile_edge_detection_results(
        cls,
        r_vals: NDArray[np.float64],
        vesicle_center: tuple[float, float],
        extraction_config: EdgeExtractionConfig,
    ) -> EdgeDetection:
        """Build a successful detection from raw extractor output."""
        center = (vesicle_center[1], vesicle_center[0])
        full_contour = ImageContour(center, r_vals)
        if extraction_config.n_angular_samples is None:
            analysis_contour = full_contour
            analysis_radii = r_vals
        else:
            analysis_radii = cls._downsample_r_vals(
                r_vals,
                extraction_config.n_angular_samples,
            )
            analysis_contour = ImageContour(center, analysis_radii)

        return EdgeDetection(
            full_contour,
            analysis_contour,
            analysis_radii / extraction_config.pixels_per_micron,
        )

    @staticmethod
    def _downsample_r_vals(
        r_vals: NDArray[np.float64],
        n_samples: int = 120,
    ) -> NDArray[np.float64]:
        """Downsample a periodic radial profile to fixed angular sampling."""
        if n_samples > r_vals.shape[0]:
            raise IndexError(
                f"Cannot downsample r_vals with len {r_vals.shape[0]} "
                f"to {n_samples}."
            )
        if n_samples == r_vals.shape[0]:
            return r_vals

        zero_to_ntheta = np.linspace(0, n_samples - 1, n_samples)
        new_evenly_spaced_indices = (
            zero_to_ntheta * (r_vals.shape[0] / n_samples)
        )
        return downsample_to_new_indices(
            r_vals,
            new_evenly_spaced_indices,
        )
