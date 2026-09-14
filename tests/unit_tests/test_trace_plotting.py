"""Tests for VesEdge contour trace overlays."""

import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from vesmod.VesEdge import (
    EdgeDetection,
    EdgeExtractionConfig,
    EdgeTracePlotConfig,
    ImageContour,
    VesicleEdges,
    VesicleQCConfig,
    VesicleQCResult,
    centered_contour_coordinates,
    contour_center_of_mass,
    plot_edge_traces,
    select_trace_detections,
)
from vesmod.cli import trace_plot_cli


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
    edges.qc_result = VesicleQCResult(config=VesicleQCConfig(curvature_threshold=1))
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


def test_centered_contour_coordinates_remove_frame_to_frame_translation():
    """Independent centering removes rigid origin translations."""
    radii = np.linspace(4.0, 6.0, 8)
    first = ImageContour((10, 20), radii)
    translated = ImageContour((410, -180), radii)

    first_xy = centered_contour_coordinates(first, 2.0)
    translated_xy = centered_contour_coordinates(translated, 2.0)

    np.testing.assert_allclose(first_xy[0], translated_xy[0], atol=1e-12)
    np.testing.assert_allclose(first_xy[1], translated_xy[1], atol=1e-12)


def test_contour_center_of_mass_is_independent_of_supplied_origin():
    """The detected contour, not its polar origin, determines its center."""
    contour = ImageContour((100, 200), np.array([10.0, 12.0, 10.0, 8.0]))

    center = contour_center_of_mass(contour)

    assert center == pytest.approx((100.0, 201.3333333333333))


def test_plot_edge_traces_with_background_anchors_to_first_center():
    """Background mode preserves native coordinates while removing translation."""
    background = np.zeros((40, 50))
    figure, axis = plot_edge_traces(
        _edges(),
        [0, 1],
        background=background,
        config=EdgeTracePlotConfig(centered=True, show_colorbar=False),
    )

    first_line, second_line = axis.lines
    assert first_line.get_xdata()[0] == pytest.approx(14)
    assert second_line.get_xdata()[0] == pytest.approx(15)
    assert axis.get_xlim() == pytest.approx((0, 50))
    assert axis.get_ylim() == pytest.approx((40, 0))
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


def test_process_trace_plot_records_checkpoint_source_path(tmp_path, monkeypatch):
    """Centered plots retain source provenance in their JSON sidecar."""
    checkpoint = tmp_path / "sample.npz"
    checkpoint.touch()
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir()
    output = tmp_path / "figures" / "traces.png"
    edges = SimpleNamespace(
        source_path=tmp_path / "sample.nd2",
        extraction_config=SimpleNamespace(pixels_per_micron=2.0),
    )
    selection = SimpleNamespace(requires_frames=False)

    monkeypatch.setattr(
        trace_plot_cli.VesicleEdges,
        "from_checkpoint",
        lambda path: edges,
    )
    monkeypatch.setattr(
        trace_plot_cli,
        "load_recorded_qc",
        lambda path, checkpoints: selection,
    )
    monkeypatch.setattr(
        trace_plot_cli,
        "replay_recorded_qc",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        trace_plot_cli,
        "plot_edge_traces",
        lambda *args, **kwargs: _blank_figure(),
    )
    args = SimpleNamespace(
        checkpoint=checkpoint,
        qc_dir=qc_dir,
        frames=[0],
        output=output,
        background_frame=None,
        contour="full",
        cmap="viridis",
        linewidth=1.5,
        alpha=1.0,
        no_colorbar=True,
        overwrite=False,
    )

    _, sidecar = trace_plot_cli.process_trace_plot(args)

    assert json.loads(sidecar.read_text())["source_path"] == str(edges.source_path)


def _blank_figure():
    """Return a minimal figure for CLI output tests."""
    import matplotlib.pyplot as plt

    return plt.subplots()
