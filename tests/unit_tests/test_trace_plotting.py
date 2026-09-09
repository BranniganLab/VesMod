"""Tests for VesEdge contour trace overlays."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from vesmod.VesEdge import (
    EdgeDetection,
    EdgeExtractionConfig,
    EdgeTracePlotConfig,
    EdgeQCConfig,
    ImageContour,
    VesicleEdges,
    VesicleQCResult,
    plot_edge_traces,
    select_trace_detections,
)


def _edges():
    detections = [
        EdgeDetection(
            ImageContour((10 + index, 20), np.full(8, 4 + index)),
            ImageContour((10 + index, 20), np.full(4, 4 + index)),
            frame_index=index,
        )
        for index in range(3)
    ]
    edges = VesicleEdges(EdgeExtractionConfig(pixels_per_micron=2), detections)
    edges.qc_result = VesicleQCResult(config=EdgeQCConfig(curvature_threshold=1))
    return edges


def test_select_trace_detections_sorts_and_rejects_invalid_frames():
    """Selection uses source indices and returns chronological detections."""
    edges = _edges()

    assert [d.frame_index for d in select_trace_detections(edges, [2, 0])] == [0, 2]
    with pytest.raises(ValueError, match="duplicates"):
        select_trace_detections(edges, [1, 1])
    with pytest.raises(IndexError):
        select_trace_detections(edges, [3])


def test_plot_edge_traces_centers_and_scales_contours():
    """White-background mode uses centered physical coordinates."""
    figure, axis = plot_edge_traces(
        _edges(),
        [0],
        config=EdgeTracePlotConfig(centered=True, show_colorbar=False),
    )

    line = axis.lines[0]
    assert np.max(np.abs(line.get_xdata())) == pytest.approx(2)
    assert np.max(np.abs(line.get_ydata())) == pytest.approx(2)
    assert axis.get_xlabel() == "x (microns)"
    figure.clf()


def test_plot_edge_traces_aligns_background_in_pixel_coordinates():
    """Background mode retains image coordinates and adds the image first."""
    background = np.zeros((40, 50))
    figure, axis = plot_edge_traces(
        _edges(),
        [0],
        background=background,
        config=EdgeTracePlotConfig(centered=False, show_colorbar=False),
    )

    line = axis.lines[0]
    assert line.get_xdata()[0] == pytest.approx(14)
    assert line.get_ydata()[0] == pytest.approx(20)
    assert axis.get_xlim() == (0, 50)
    assert axis.get_ylim() == (40, 0)
    figure.clf()


def test_plot_edge_traces_rejects_centered_background():
    """A centered contour cannot be overlaid on native image coordinates."""
    with pytest.raises(ValueError, match="centered=False"):
        plot_edge_traces(
            _edges(),
            [0],
            background=np.zeros((10, 10)),
            config=EdgeTracePlotConfig(centered=True),
        )
