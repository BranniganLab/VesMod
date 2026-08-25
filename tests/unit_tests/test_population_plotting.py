"""Unit tests for population-QC diagnostic plotting."""

from matplotlib.axes import Axes
import numpy as np
import pytest

from vesmod.VesEdge.models import EdgeDetection, ImageContour, QCFlag
from vesmod.VesEdge.population_plotting import save_population_histograms


def _make_assigned_edge(origin, radius, population_label):
    """Return one edge with a completed population assignment."""
    radii = np.full(8, radius, dtype=float)
    edge = EdgeDetection(
        ImageContour(origin, radii.copy()),
        ImageContour(origin, radii.copy()),
    )
    edge.qc.population_label = population_label
    edge.qc.population_probability = 1.0
    return edge


def test_save_population_histograms_creates_labeled_radius_figure(
    tmp_path,
    monkeypatch,
):
    """Test each fitted population is represented in the radius histogram."""
    detections = [
        _make_assigned_edge((0.0, 1.0), 10.0, 0),
        _make_assigned_edge((0.5, 1.5), 11.0, 0),
        _make_assigned_edge((20.0, 21.0), 3.0, 1),
    ]
    labels = []
    original_hist = Axes.hist

    def record_hist(axis, values, *args, **kwargs):
        labels.append(kwargs["label"])
        return original_hist(axis, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", record_hist)
    output_path = tmp_path / "sample.population_histograms.png"

    save_population_histograms(detections, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert labels.count("Population 0 (n=2)") == 1
    assert labels.count("Population 1 (n=1)") == 1


def test_save_population_histograms_requires_population_assignments(tmp_path):
    """Test plotting before population QC fails with a clear error."""
    radii = np.full(8, 10.0, dtype=float)
    edge = EdgeDetection(
        ImageContour((0.0, 0.0), radii),
        ImageContour((0.0, 0.0), radii),
    )

    with pytest.raises(ValueError, match="Population QC must assign"):
        save_population_histograms([edge], tmp_path / "histogram.png")


def test_save_population_histograms_skips_when_preceding_qc_rejects_all(tmp_path):
    """Test zero population-eligible detections produce no figure or error."""
    radii = np.full(8, 10.0, dtype=float)
    edge = EdgeDetection(
        ImageContour((0.0, 0.0), radii),
        ImageContour((0.0, 0.0), radii),
    )
    edge.qc.flags.add(QCFlag.CURVATURE)
    output_path = tmp_path / "histogram.png"

    save_population_histograms([edge], output_path)

    assert not output_path.exists()
