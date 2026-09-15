"""Composable Matplotlib animation helpers for VesEdge diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
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
    frame_indices: Sequence[int] | None = None

    def __post_init__(self) -> None:
        """Resolve and validate source frames represented by the animation."""
        _validate_vesicle_edges(self.video, self.edges)
        if self.frame_indices is None:
            self.frame_indices = tuple(range(self.video.frames.shape[0]))
            return
        try:
            indices = tuple(self.frame_indices)
        except TypeError as error:
            raise TypeError("frame_indices must be an iterable of integers.") from error
        if not indices:
            raise ValueError("frame_indices must contain at least one frame index.")
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError(
                    "frame_indices must contain only integer frame indices."
                )
            if index < 0 or index >= self.video.frames.shape[0]:
                raise IndexError(
                    f"frame index {index} is outside the range "
                    f"0..{self.video.frames.shape[0] - 1}."
                )
        self.frame_indices = tuple(int(index) for index in indices)

    @property
    def n_frames(self) -> int:
        """Return the number of selected source frames."""
        return len(self.frame_indices)

    def draw(self, ax: Axes, frame_index: int) -> None:
        """Draw one selected vesicle frame on the supplied axes."""
        if frame_index < 0 or frame_index >= self.n_frames:
            raise IndexError(
                f"animation frame index {frame_index} is outside the range "
                f"0..{self.n_frames - 1}."
            )
        source_frame_index = self.frame_indices[frame_index]
        draw_vesicle_frame(
            self.video,
            ax,
            source_frame_index,
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
    layout: tuple[int, int] | None = None,
) -> None:
    """Save synchronized animation panels as a single GIF.

    Parameters
    ----------
    path : str | Path
        Output GIF path.
    panels : list[AnimationPanel]
        Panels arranged in row-major order and advanced with a shared frame
        index. Every panel must expose the same number of frames.
    interval : int
        Delay between frames in milliseconds.
    repeat_delay : int
        Delay before the animation repeats in milliseconds.
    figsize : tuple[float, float] | None
        Optional Matplotlib figure size in inches.
    layout : tuple[int, int] | None
        Number of (rows, columns) in the figure. When omitted, use one row
        with one column per panel. The layout must provide exactly one slot
        for each panel.

    Example
    -------
    Arrange four panels for an old-versus-new extraction comparison::

        panels = [
            VesicleAnimationPanel(video_a, old_edges_a),
            VesicleAnimationPanel(video_a, new_edges_a),
            VesicleAnimationPanel(video_b, old_edges_b),
            VesicleAnimationPanel(video_b, new_edges_b),
        ]
        make_gif("comparison.gif", panels, layout=(2, 2))

    Custom panels can transform contours in their draw method while
    continuing to use this controller for shared frame indices and layout.

    Raises
    ------
    ValueError
        If no panels are supplied, the layout is invalid, a panel has fewer
        than one frame, or panel frame counts differ.
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

    if layout is None:
        rows, columns = 1, len(panels)
    elif (
        not isinstance(layout, tuple)
        or len(layout) != 2
        or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
            for dimension in layout
        )
    ):
        raise ValueError(
            "layout must be a pair of positive integers (rows, columns)."
        )
    else:
        rows, columns = layout

    if rows * columns != len(panels):
        raise ValueError(
            "The layout must contain exactly one slot per animation panel; "
            f"received layout {rows}x{columns} for {len(panels)} panels."
        )

    output_path = Path(path).with_suffix(".gif")
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    panel_axes: NDArray[np.object_] = axes.reshape(-1)

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
    frame_indices: Sequence[int] | None = None,
) -> None:
    """Save a single-panel vesicle GIF using the composable animation API.

    ``frame_indices`` optionally selects source frames, in the supplied order.
    Edge rendering, decorators, and title providers receive the corresponding
    source-frame index rather than the animation-frame index.
    """
    _validate_vesicle_edges(video, edges)
    panel = VesicleAnimationPanel(
        video,
        edges=edges,
        frame_decorator=frame_decorator,
        title_provider=title_provider,
        frame_indices=frame_indices,
    )
    make_gif(path, [panel])
