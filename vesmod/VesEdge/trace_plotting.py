"""Plot time-ordered overlays of extracted VesEdge contours."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from numpy.typing import NDArray

from .models import EdgeDetection, ImageContour
from .vesicle_edges import VesicleEdges


@dataclass(frozen=True)
class EdgeTracePlotConfig:
    """Rendering options for :func:`plot_edge_traces`."""

    centered: bool = True
    contour: str = "full"
    cmap: str = "viridis"
    linewidth: float = 1.5
    alpha: float = 1.0
    show_colorbar: bool = True

    def __post_init__(self) -> None:
        """Validate plotting options early."""
        if self.contour not in {"full", "analysis"}:
            raise ValueError("contour must be 'full' or 'analysis'.")
        if self.linewidth <= 0:
            raise ValueError("linewidth must be positive.")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be greater than 0 and at most 1.")


def centered_contour_coordinates(
    contour: ImageContour,
    pixels_per_micron: float,
    reference_center: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a contour centered on a reference point and scaled to microns.

    ``ImageContour.x`` and ``ImageContour.y`` are expressed in source-image
    coordinates and therefore include the supplied polar origin. By default,
    the contour is centered on its detected-edge center of mass.
    ``reference_center`` can be supplied to express it relative to a shared
    center, such as the first frame in a trace series. The returned arrays are
    closed for direct plotting.
    """
    scale = float(pixels_per_micron)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("pixels_per_micron must be finite and positive.")
    center = (
        contour_center_of_mass(contour)
        if reference_center is None
        else reference_center
    )
    x = (contour.x - center[0]) / scale
    y = (contour.y - center[1]) / scale
    return x, y


def contour_center_of_mass(contour: ImageContour) -> tuple[float, float]:
    """Return the uniform-area center of mass enclosed by a contour.

    The calculation uses the polygon formed by the detected edge, rather than
    the origin used to construct its polar representation.
    """
    x = contour.x[:-1]
    y = contour.y[:-1]
    next_x = np.roll(x, -1)
    next_y = np.roll(y, -1)
    cross = x * next_y - next_x * y
    twice_area = float(np.sum(cross))
    if np.isclose(twice_area, 0.0):
        raise ValueError("The contour must enclose a nonzero area.")
    center_x = float(np.sum((x + next_x) * cross) / (3.0 * twice_area))
    center_y = float(np.sum((y + next_y) * cross) / (3.0 * twice_area))
    return center_x, center_y


def select_trace_detections(
    edges: VesicleEdges,
    frame_indices: Sequence[int],
) -> list[EdgeDetection]:
    """Return QC-accepted detections for sorted source-frame indices.

    Frame indices refer to original source-video frames, including frames that
    failed extraction. The returned detections are always chronological.
    """
    if edges.qc_result is None:
        raise ValueError("Quality control must be run before plotting traces.")
    if not edges.qc_result.passed:
        raise ValueError("The vesicle trajectory failed quality control.")

    indices = list(frame_indices)
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        for index in indices
    ):
        raise TypeError("frame_indices must contain integers.")
    indices = [int(index) for index in indices]
    if not indices:
        raise ValueError("At least one frame index is required.")
    if len(set(indices)) != len(indices):
        raise ValueError("frame_indices must not contain duplicates.")
    if any(index < 0 for index in indices):
        raise ValueError("frame_indices must be nonnegative.")
    if any(index >= len(edges.detections) for index in indices):
        raise IndexError(
            f"frame index must be between 0 and {len(edges.detections) - 1}."
        )

    selected: list[EdgeDetection] = []
    for index in sorted(indices):
        result = edges.detections[index]
        if not isinstance(result, EdgeDetection):
            raise ValueError(f"Source frame {index} has no extracted contour.")
        if not result.qc.passed:
            raise ValueError(f"Source frame {index} was rejected by QC.")
        selected.append(result)
    return selected


def plot_edge_traces(
    edges: VesicleEdges,
    frame_indices: Sequence[int],
    *,
    ax=None,
    background: NDArray[np.number] | None = None,
    config: EdgeTracePlotConfig | None = None,
    centered: bool | None = None,
    contour: str | None = None,
    cmap: str | None = None,
    linewidth: float | None = None,
    alpha: float | None = None,
    show_colorbar: bool | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot selected contours with color indicating source-frame time.

    With ``centered=True`` contours are centered independently on their
    detected-edge centers of mass and converted to microns when no background
    is supplied. With a background, contours are translated so their
    detected-edge centers of mass remain at the first selected contour's
    center, while the background remains in native source-image pixel
    coordinates.
    With ``centered=False`` contours remain in raw source-image pixel
    coordinates without stabilization.
    """
    if config is None:
        config = EdgeTracePlotConfig()
    overrides = {
        "centered": centered,
        "contour": contour,
        "cmap": cmap,
        "linewidth": linewidth,
        "alpha": alpha,
        "show_colorbar": show_colorbar,
    }
    if any(value is not None for value in overrides.values()):
        config = replace(
            config,
            **{
                name: value
                for name, value in overrides.items()
                if value is not None
            },
        )
    detections = select_trace_detections(edges, frame_indices)
    requested_indices = [detection.frame_index for detection in detections]
    if any(index is None for index in requested_indices):
        raise ValueError("Selected detections must have source frame indices.")
    source_indices = np.asarray(requested_indices, dtype=float)
    norm = Normalize(
        vmin=float(source_indices.min()),
        vmax=float(source_indices.max()),
    )
    if norm.vmin == norm.vmax:
        norm = Normalize(vmin=norm.vmin - 0.5, vmax=norm.vmax + 0.5)
    colormap = plt.get_cmap(config.cmap)

    if ax is None:
        figure, axis = plt.subplots()
    else:
        axis = ax
        figure = axis.figure

    contour_centers = (
        {
            id(detection): contour_center_of_mass(
                detection.full_contour
                if config.contour == "full"
                else detection.analysis_contour
            )
            for detection in detections
        }
        if config.centered
        else {}
    )
    reference_center = None
    if background is not None and config.centered:
        reference_center = contour_centers[id(detections[0])]
        axis.imshow(background, origin="upper", cmap="gray")
    elif background is not None:
        axis.imshow(background, origin="upper", cmap="gray")

    units = "pixels"
    for detection in detections:
        contour = (
            detection.full_contour
            if config.contour == "full"
            else detection.analysis_contour
        )
        if config.centered:
            detected_center = contour_centers[id(detection)]
            x, y = centered_contour_coordinates(
                contour,
                1.0
                if background is not None
                else edges.extraction_config.pixels_per_micron,
                reference_center=detected_center,
            )
            if reference_center is not None:
                x += reference_center[0]
                y += reference_center[1]
            units = "pixels" if background is not None else "microns"
        else:
            x = contour.x
            y = contour.y
        frame_value = float(detection.frame_index)
        axis.plot(
            x,
            y,
            color=colormap(norm(frame_value)),
            linewidth=config.linewidth,
            alpha=config.alpha,
        )

    if background is not None:
        axis.set_xlim(0, background.shape[1])
        axis.set_ylim(background.shape[0], 0)
        axis.set_aspect("equal")
        axis.set_xlabel("x (pixels)")
        axis.set_ylabel("y (pixels)")
    else:
        axis.set_aspect("equal")
        axis.set_xlabel(f"x ({units})")
        axis.set_ylabel(f"y ({units})")

    if config.show_colorbar:
        scalar_mappable = ScalarMappable(norm=norm, cmap=colormap)
        scalar_mappable.set_array(source_indices)
        figure.colorbar(scalar_mappable, ax=axis, label="Source frame")
    return figure, axis
