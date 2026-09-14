"""Composable Matplotlib animation helpers for VesEdge diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import EdgeDetection

if TYPE_CHECKING:
    from .vesicle_edges import VesicleEdges
    from .vesicle_video import VesicleVideo


class AnimationPanel(Protocol):
    """Interface implemented by one panel in a synchronized animation."""

    @property
    def n_frames(self) -> int:
        """Return the number of animation frames available to the panel."""

    def draw(self, ax: Axes, frame_index: int) -> None:
        """Draw one animation frame onto the supplied axes."""


def _validate_vesicle_edges(
    video: VesicleVideo,
    edges: VesicleEdges | None,
) -> None:
    """Verify optional edge results correspond one-to-one with video frames."""
    if edges is not None and len(edges.detections) != video.frames.shape[0]:
        raise ValueError(
            f"There are {len(edges.detections)} detections and "
            f"{video.frames.shape[0]} frames."
        )


def draw_vesicle_frame(
    video: VesicleVideo,
    ax: Axes,
    frame_index: int,
    edges: VesicleEdges | None = None,
    frame_decorator: Callable[[Axes, int], None] | None = None,
    title_provider: Callable[[int], str] | None = None,
) -> None:
    """Draw one vesicle video frame and optional edge/QC overlays.

    Parameters
    ----------
    video : VesicleVideo
        Source video providing the image frame.
    ax : matplotlib.axes.Axes
        Axes on which to draw the requested frame.
    frame_index : int
        Zero-based source-frame index.
    edges : VesicleEdges | None
        Optional detections to draw with standard QC-aware coloring.
    frame_decorator : Callable[[matplotlib.axes.Axes, int], None] | None
        Optional callback that decorates the axes after the image and detected
        edge are drawn.
    title_provider : Callable[[int], str] | None
        Optional callback returning the title for the frame. Without one, the
        renderer uses the default frame-number title.

    Raises
    ------
    ValueError
        If supplied extraction results do not match the frame count.
    IndexError
        If ``frame_index`` is outside the video frame range.
    """
    _validate_vesicle_edges(video, edges)
    if frame_index < 0 or frame_index >= video.frames.shape[0]:
        raise IndexError(
            f"Frame index {frame_index} is outside the range "
            f"0..{video.frames.shape[0] - 1}."
        )

    ax.clear()
    ax.imshow(video.frames[frame_index], cmap="gray", animated=True)
    if edges is not None:
        result = edges.detections[frame_index]
        if isinstance(result, EdgeDetection):
            contour = result.full_contour
            color = "tab:green"
            if edges.qc_result is not None and (
                not edges.qc_result.passed or not result.qc.passed
            ):
                color = "tab:red"
            ax.plot(contour.x, contour.y, color=color)
    if frame_decorator is not None:
        frame_decorator(ax, frame_index)
    frame_title = f"frame {frame_index} / {video.frames.shape[0]}"
    if title_provider is not None:
        frame_title = title_provider(frame_index)
    ax.set_title(frame_title)


@dataclass
class VesicleAnimationPanel:
    """Render vesicle video frames with optional edge and QC overlays."""

    video: VesicleVideo
    edges: VesicleEdges | None = None
    frame_decorator: Callable[[Axes, int], None] | None = None
    title_provider: Callable[[int], str] | None = None

    @property
    def n_frames(self) -> int:
        """Return the number of source video frames."""
        return self.video.frames.shape[0]

    def draw(self, ax: Axes, frame_index: int) -> None:
        """Draw one vesicle frame on the supplied axes."""
        draw_vesicle_frame(
            self.video,
            ax,
            frame_index,
            self.edges,
            frame_decorator=self.frame_decorator,
            title_provider=self.title_provider,
        )


@dataclass
class TimeSeriesAnimationPanel:
    """Render a time series with a marker at the current animation frame."""

    time: ArrayLike
    values: ArrayLike
    ylabel: str | None = None
    xlabel: str = "Time"
    title: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize time-series data."""
        self.time = np.asarray(self.time)
        self.values = np.asarray(self.values)
        if self.time.ndim != 1 or self.values.ndim != 1:
            raise ValueError("time and values must both be one-dimensional.")
        if self.time.shape[0] != self.values.shape[0]:
            raise ValueError(
                "time and values must contain the same number of samples."
            )
        if self.time.shape[0] == 0:
            raise ValueError("time and values must contain at least one sample.")

    @property
    def n_frames(self) -> int:
        """Return the number of time-series samples."""
        return self.time.shape[0]

    def draw(self, ax: Axes, frame_index: int) -> None:
        """Draw the full trace and highlight the current sample."""
        if frame_index < 0 or frame_index >= self.n_frames:
            raise IndexError(
                f"Frame index {frame_index} is outside the range "
                f"0..{self.n_frames - 1}."
            )
        ax.clear()
        ax.plot(self.time, self.values)
        ax.plot(
            self.time[frame_index],
            self.values[frame_index],
            marker="o",
        )
        ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        if self.title is not None:
            ax.set_title(self.title)


def make_gif(
    path: str | Path,
    panels: list[AnimationPanel],
    *,
    interval: int = 150,
    repeat_delay: int = 1000,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Save synchronized animation panels as a single GIF.

    Parameters
    ----------
    path : str | Path
        Output GIF path.
    panels : list[AnimationPanel]
        Panels drawn side by side and advanced with a shared frame index.
        Every panel must expose the same number of frames.
    interval : int
        Delay between frames in milliseconds.
    repeat_delay : int
        Delay before the animation repeats in milliseconds.
    figsize : tuple[float, float] | None
        Optional Matplotlib figure size in inches.

    Raises
    ------
    ValueError
        If no panels are supplied, a panel has fewer than one frame, or panel
        frame counts differ.
    TypeError
        If an object does not provide ``n_frames`` and ``draw``.
    """
    if not panels:
        raise ValueError("At least one animation panel is required.")

    frame_counts = []
    for panel in panels:
        if not hasattr(panel, "n_frames") or not callable(
            getattr(panel, "draw", None)
        ):
            raise TypeError(
                "Every panel must define n_frames and draw(ax, frame_index)."
            )
        frame_counts.append(panel.n_frames)

    if any(frame_count < 1 for frame_count in frame_counts):
        raise ValueError(
            "Every animation panel must contain at least one frame; "
            f"received {frame_counts}."
        )
    if len(set(frame_counts)) != 1:
        raise ValueError(
            "All animation panels must contain the same number of frames; "
            f"received {frame_counts}."
        )

    output_path = Path(path).with_suffix(".gif")
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
    panel_axes: NDArray[np.object_] = axes[0]

    def animate(frame_index: int) -> None:
        for panel, ax in zip(panels, panel_axes):
            panel.draw(ax, frame_index)

    try:
        animation = FuncAnimation(
            fig,
            animate,
            frames=frame_counts[0],
            interval=interval,
            blit=False,
            repeat_delay=repeat_delay,
        )
        animation.save(output_path)
    finally:
        plt.close(fig)


def make_vesicle_gif(
    video: VesicleVideo,
    path: str | Path,
    edges: VesicleEdges | None = None,
    frame_decorator: Callable[[Axes, int], None] | None = None,
    title_provider: Callable[[int], str] | None = None,
) -> None:
    """Save a single-panel vesicle GIF using the composable animation API."""
    _validate_vesicle_edges(video, edges)
    panel = VesicleAnimationPanel(
        video,
        edges=edges,
        frame_decorator=frame_decorator,
        title_provider=title_provider,
    )
    make_gif(path, [panel])
