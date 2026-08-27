"""Unit tests for composable VesEdge animation helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vesmod.VesEdge import (
    TimeSeriesAnimationPanel,
    VesicleAnimationPanel,
    VesicleVideo,
    make_gif,
)


def test_vesicle_animation_panel_delegates_to_video_draw_frame(monkeypatch):
    """Test vesicle panels render through the video axes primitive."""
    video = VesicleVideo(np.zeros((3, 10, 10)))
    observed = []

    def fake_draw_frame(axis, frame_index, edges=None, **kwargs):
        observed.append((axis, frame_index, edges, kwargs))

    monkeypatch.setattr(video, "draw_frame", fake_draw_frame)
    panel = VesicleAnimationPanel(video)
    fig, ax = plt.subplots()

    panel.draw(ax, 2)

    assert panel.n_frames == 3
    assert observed == [
        (
            ax,
            2,
            None,
            {"frame_decorator": None, "title_provider": None},
        )
    ]
    plt.close(fig)


def test_time_series_panel_marks_current_sample():
    """Test time-series panels plot the trace and current-frame marker."""
    panel = TimeSeriesAnimationPanel(
        [0.0, 1.0, 2.0],
        [10.0, 20.0, 15.0],
        xlabel="Time (s)",
        ylabel="Area",
        title="Area over time",
    )
    fig, ax = plt.subplots()

    panel.draw(ax, 1)

    assert panel.n_frames == 3
    assert np.array_equal(ax.lines[0].get_xdata(), [0.0, 1.0, 2.0])
    assert np.array_equal(ax.lines[0].get_ydata(), [10.0, 20.0, 15.0])
    assert np.array_equal(ax.lines[1].get_xdata(), [1.0])
    assert np.array_equal(ax.lines[1].get_ydata(), [20.0])
    assert ax.get_xlabel() == "Time (s)"
    assert ax.get_ylabel() == "Area"
    assert ax.get_title() == "Area over time"
    plt.close(fig)


def test_time_series_panel_requires_matching_one_dimensional_data():
    """Test time-series panel input validation."""
    with pytest.raises(ValueError, match="same number"):
        TimeSeriesAnimationPanel([0, 1], [1])
    with pytest.raises(ValueError, match="one-dimensional"):
        TimeSeriesAnimationPanel([[0, 1]], [[1, 2]])
    with pytest.raises(ValueError, match="at least one"):
        TimeSeriesAnimationPanel([], [])


def test_make_gif_draws_synchronized_panels(monkeypatch, tmp_path):
    """Test the generic animator advances every panel with one frame index."""
    observed = []

    class FakePanel:
        n_frames = 3

        def __init__(self, name):
            self.name = name

        def draw(self, axis, frame_index):
            observed.append((self.name, axis, frame_index))

    class FakeAnimation:
        def __init__(self, _, animate, frames, **__):
            for frame_index in range(frames):
                animate(frame_index)

        @staticmethod
        def save(_):
            return None

    monkeypatch.setattr(
        "vesmod.VesEdge.animation.FuncAnimation",
        FakeAnimation,
    )
    panels = [FakePanel("vesicle"), FakePanel("area")]

    make_gif(tmp_path / "combined.gif", panels)

    assert [(name, index) for name, _, index in observed] == [
        ("vesicle", 0),
        ("area", 0),
        ("vesicle", 1),
        ("area", 1),
        ("vesicle", 2),
        ("area", 2),
    ]
    assert observed[0][1] is observed[2][1]
    assert observed[1][1] is observed[3][1]
    assert observed[0][1] is not observed[1][1]


def test_make_gif_rejects_mismatched_frame_counts(tmp_path):
    """Test synchronized panels must have the same number of frames."""
    class FakePanel:
        def __init__(self, n_frames):
            self.n_frames = n_frames

        @staticmethod
        def draw(axis, frame_index):
            return None

    with pytest.raises(ValueError, match="same number of frames"):
        make_gif(
            tmp_path / "combined.gif",
            [FakePanel(2), FakePanel(3)],
        )


def test_make_gif_requires_at_least_one_panel(tmp_path):
    """Test an animation cannot be constructed without panels."""
    with pytest.raises(ValueError, match="At least one"):
        make_gif(tmp_path / "empty.gif", [])
