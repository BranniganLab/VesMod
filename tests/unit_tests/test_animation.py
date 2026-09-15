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


def test_vesicle_animation_panel_delegates_to_frame_renderer(monkeypatch):
    """Test vesicle panels render through the standalone axes primitive."""
    video = VesicleVideo(np.zeros((3, 10, 10)))
    observed = []

    def fake_draw_vesicle_frame(video_arg, axis, frame_index, edges=None, **kwargs):
        observed.append((video_arg, axis, frame_index, edges, kwargs))

    monkeypatch.setattr(
        "vesmod.VesEdge.animation.draw_vesicle_frame",
        fake_draw_vesicle_frame,
    )
    panel = VesicleAnimationPanel(video)
    fig, ax = plt.subplots()

    panel.draw(ax, 2)

    assert panel.n_frames == 3
    assert observed == [
        (
            video,
            ax,
            2,
            None,
            {"frame_decorator": None, "title_provider": None},
        )
    ]
    plt.close(fig)


def test_vesicle_animation_panel_maps_selected_animation_frames(monkeypatch):
    """Test selected source frames retain their order for all callbacks."""
    video = VesicleVideo(np.zeros((3, 10, 10)))
    observed = []

    def fake_draw_vesicle_frame(video_arg, axis, frame_index, edges=None, **kwargs):
        observed.append((video_arg, axis, frame_index, edges, kwargs))

    monkeypatch.setattr(
        "vesmod.VesEdge.animation.draw_vesicle_frame",
        fake_draw_vesicle_frame,
    )
    decorator = object()
    title_provider = object()
    panel = VesicleAnimationPanel(
        video,
        frame_indices=[2, 0],
        frame_decorator=decorator,
        title_provider=title_provider,
    )
    fig, ax = plt.subplots()

    panel.draw(ax, 0)
    panel.draw(ax, 1)

    assert panel.n_frames == 2
    assert [entry[2] for entry in observed] == [2, 0]
    assert all(entry[4]["frame_decorator"] is decorator for entry in observed)
    assert all(entry[4]["title_provider"] is title_provider for entry in observed)
    plt.close(fig)


@pytest.mark.parametrize(
    ("frame_indices", "exception", "message"),
    [
        ([], ValueError, "at least one"),
        ([3], IndexError, "outside the range"),
        ([-1], IndexError, "outside the range"),
        ([True], TypeError, "integer frame indices"),
        (["1"], TypeError, "integer frame indices"),
    ],
)
def test_vesicle_animation_panel_rejects_invalid_frame_indices(
    frame_indices,
    exception,
    message,
):
    """Test invalid source-frame selections fail before animation starts."""
    video = VesicleVideo(np.zeros((3, 10, 10)))

    with pytest.raises(exception, match=message):
        VesicleAnimationPanel(video, frame_indices=frame_indices)


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


def test_make_gif_places_four_synchronized_panels_in_two_by_two_layout(
    monkeypatch, tmp_path
):
    """Test controller layout preserves one shared frame index per panel."""
    observed = []

    class FakePanel:
        n_frames = 2

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
    panel_names = ("old-a", "new-a", "old-b", "new-b")
    panels = [FakePanel(name) for name in panel_names]

    make_gif(tmp_path / "comparison.gif", panels, layout=(2, 2))

    assert [(name, index) for name, _, index in observed] == [
        (name, frame_index)
        for frame_index in range(2)
        for name in panel_names
    ]
    panel_axes = [axis for _, axis, _ in observed[:4]]
    assert len({id(axis) for axis in panel_axes}) == 4
    assert [
        axis.get_subplotspec().rowspan.start for axis in panel_axes
    ] == [0, 0, 1, 1]
    assert [
        axis.get_subplotspec().colspan.start for axis in panel_axes
    ] == [0, 1, 0, 1]


def test_make_gif_rejects_invalid_layout(tmp_path):
    """Test layouts must be positive integer pairs with one slot per panel."""
    class FakePanel:
        n_frames = 1

        @staticmethod
        def draw(axis, frame_index):
            return None

    panels = [FakePanel() for _ in range(4)]
    invalid_layouts = ((0, 2), (2, -1), (2,), (2, 2, 1), (2.0, 2), [2, 2])
    for layout in invalid_layouts:
        with pytest.raises(ValueError, match="layout"):
            make_gif(tmp_path / "invalid.gif", panels, layout=layout)

    with pytest.raises(ValueError, match="exactly one slot"):
        make_gif(tmp_path / "invalid.gif", panels[:3], layout=(2, 2))


def test_make_gif_can_draw_a_transformed_custom_panel(monkeypatch, tmp_path):
    """Test custom panels can center contours using the public panel protocol."""
    contour = np.array([[10.0, 12.0], [14.0, 18.0], [18.0, 12.0]])
    rendered = []

    class CenteredPanel:
        n_frames = 1

        def draw(self, axis, frame_index):
            centered = contour - contour.mean(axis=0)
            axis.plot(centered[:, 0], centered[:, 1])
            rendered.append(
                (
                    frame_index,
                    axis.lines[0].get_xdata(),
                    axis.lines[0].get_ydata(),
                )
            )

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

    make_gif(tmp_path / "centered.gif", [CenteredPanel()])

    assert len(rendered) == 1
    frame_index, xdata, ydata = rendered[0]
    assert frame_index == 0
    assert np.array_equal(xdata, [-4.0, 0.0, 4.0])
    assert np.array_equal(ydata, [-2.0, 4.0, -2.0])


def test_make_gif_closes_figure_when_save_raises(monkeypatch, tmp_path):
    """Test the animator closes its figure when saving fails."""
    observed = {}

    class FakePanel:
        n_frames = 1

        @staticmethod
        def draw(axis, frame_index):
            return None

    class FakeAnimation:
        def __init__(self, fig, animate, frames, **__):
            observed["fig"] = fig

        @staticmethod
        def save(_):
            raise RuntimeError("save failed")

    monkeypatch.setattr(
        "vesmod.VesEdge.animation.FuncAnimation",
        FakeAnimation,
    )

    with pytest.raises(RuntimeError, match="save failed"):
        make_gif(tmp_path / "combined.gif", [FakePanel()])

    assert not plt.fignum_exists(observed["fig"].number)


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


def test_make_gif_rejects_zero_frame_video(tmp_path):
    """Test zero-frame videos fail before animation or saving begins."""
    video = VesicleVideo(np.empty((0, 10, 10)))
    panel = VesicleAnimationPanel(video)

    with pytest.raises(ValueError, match="at least one frame"):
        make_gif(tmp_path / "empty.gif", [panel])


def test_make_gif_requires_at_least_one_panel(tmp_path):
    """Test an animation cannot be constructed without panels."""
    with pytest.raises(ValueError, match="At least one"):
        make_gif(tmp_path / "empty.gif", [])
