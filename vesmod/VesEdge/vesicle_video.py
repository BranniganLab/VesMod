#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw vesicle video data and edge-extraction orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from .config import EdgeExtractionConfig
from .models import (
    EdgeDetection,
    EdgeDetectionFailure,
    EdgeResult,
    ImageContour,
)
from .vesicle_edges import VesicleEdges
from .vesicle_video_utils import downsample_to_new_indices


@dataclass
class VesicleVideo:
    """Raw image frames for one vesicle trajectory.

    Parameters
    ----------
    frames : np.ndarray
        Three-dimensional array containing the source image frames.
    source_path : str | Path | None
        Path to the original source video, when known.
    """

    frames: np.ndarray
    source_path: str | Path | None = None

    def __post_init__(self) -> None:
        """Validate raw image frames and source provenance."""
        if not isinstance(self.frames, np.ndarray):
            raise TypeError("frames must be a numpy ndarray.")
        if self.frames.ndim != 3:
            raise IndexError("frames must be a 3D array.")
        if self.source_path is not None:
            self.source_path = Path(self.source_path)

    def extract_edges(
        self,
        extractor_func: Callable[
            [NDArray[np.float64]],
            tuple[NDArray[np.float64], tuple[float, float]],
        ],
        extraction_config: EdgeExtractionConfig,
    ) -> VesicleEdges:
        """Extract an edge from each frame without running quality control.

        Per-frame extractor exceptions are recorded as
        :class:`EdgeDetectionFailure` so later frames can still be processed.

        Raises
        ------
        ValueError
            If no frame produces a successful detection or successful analysis
            contours have inconsistent lengths.
        IndexError
            If the configured analysis sample count exceeds the number of
            samples returned by the extractor for a frame.
        """
        detections: list[EdgeResult] = []
        for frame_index, frame in enumerate(self.frames):
            try:
                r_vals, vesicle_center = extractor_func(frame)
                self._validate_extractor_results(r_vals)
            # Extractors are user-provided callables; any per-frame failure
            # should be recorded without aborting the remaining trajectory.
            # pylint: disable-next=broad-exception-caught
            except Exception as error:  # noqa: BLE001
                detections.append(
                    EdgeDetectionFailure(
                        str(error),
                        frame_index=frame_index,
                    )
                )
                continue

            try:
                detected_edge = self._compile_edge_detection_results(
                    r_vals,
                    vesicle_center,
                    extraction_config,
                    frame_index=frame_index,
                )
            except IndexError as error:
                n_samples = extraction_config.n_angular_samples
                if n_samples is not None and n_samples > r_vals.shape[0]:
                    raise
                detections.append(
                    EdgeDetectionFailure(
                        str(error),
                        frame_index=frame_index,
                    )
                )
                continue
            # Compilation can still fail on malformed extractor output; retain
            # the established per-frame failure behavior for those cases.
            # pylint: disable-next=broad-exception-caught
            except Exception as error:  # noqa: BLE001
                detections.append(
                    EdgeDetectionFailure(
                        str(error),
                        frame_index=frame_index,
                    )
                )
                continue

            detections.append(detected_edge)

        successful_count = sum(
            isinstance(result, EdgeDetection)
            for result in detections
        )
        if successful_count == 0:
            errors = [
                result.error
                for result in detections
                if isinstance(result, EdgeDetectionFailure)
            ]
            error_summary = "; ".join(errors)
            raise ValueError(
                "Edge extraction produced no successful detections. "
                f"Extractor errors: {error_summary}"
            )

        return VesicleEdges(
            extraction_config=extraction_config,
            detections=detections,
            source_path=self.source_path,
        )

    def make_vesicle_gif(
        self,
        path: str | Path,
        edges: VesicleEdges | None = None,
        frame_overlay: Callable[[Axes, int], str | None] | None = None,
    ) -> None:
        """Save a GIF with optional edge and frame-specific overlays.

        Successful detections are green before QC and after passing QC, and
        red after a completed QC run rejects them.

        Parameters
        ----------
        path : str | Path
            Output GIF path.
        edges : VesicleEdges | None
            Optional detections to draw with standard QC-aware coloring.
        frame_overlay : Callable[[matplotlib.axes.Axes, int], str | None] | None
            Optional callback that adds analysis-specific artists for a frame.
            A returned string replaces the default frame title. The callback
            does not control edge rendering or QC colors.

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
            ax.imshow(self.frames[index], cmap="gray", animated=True)
            if edges is not None:
                result = edges.detections[index]
                if isinstance(result, EdgeDetection):
                    contour = result.full_contour
                    color = "tab:green"
                    if edges.qc_config is not None and not result.qc.passed:
                        color = "tab:red"
                    ax.plot(contour.x, contour.y, color=color)
            title = None
            if frame_overlay is not None:
                title = frame_overlay(ax, index)
            ax.set_title(title or f"frame {index} / {self.frames.shape[0]}")

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
        *,
        frame_index: int | None = None,
    ) -> EdgeDetection:
        """Build a successful detection from raw extractor output."""
        center = (vesicle_center[1], vesicle_center[0])
        full_contour = ImageContour(center, r_vals)
        if extraction_config.n_angular_samples is None:
            analysis_contour = full_contour
        else:
            analysis_radii = cls._downsample_r_vals(
                r_vals,
                extraction_config.n_angular_samples,
            )
            analysis_contour = ImageContour(center, analysis_radii)

        return EdgeDetection(
            full_contour,
            analysis_contour,
            frame_index=frame_index,
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
