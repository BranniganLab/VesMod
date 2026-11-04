"""Unit tests for localized-deviation frame QC."""

import numpy as np
import pytest

from vesmod.VesEdge.experimental.area_qc import AreaQCConfig
from vesmod.VesEdge.qc.checks.edge_filtering import CurvatureQCConfig
from vesmod.VesEdge.models import EdgeDetection, ImageContour
from vesmod.VesEdge.extraction.config import EdgeExtractionConfig
from vesmod.VesEdge.qc.checks.localized_deviation_qc import (
    LocalizedDeviationQCConfig,
    check_localized_deviation,
)
from vesmod.VesEdge.vesicle_edges import VesicleEdges
from vesmod.VesEdge.qc.qc_config import VesicleQCConfig


def edge(radii):
    """Build a detection with identical native and analysis contours."""
    contour = ImageContour((10.0, 10.0), np.asarray(radii, dtype=float))
    return EdgeDetection(contour, contour)


def qc_config():
    """Return a config with only localized-deviation QC enabled."""
    return VesicleQCConfig(
        curvature=CurvatureQCConfig(0.0, enabled=False),
        area=AreaQCConfig(enabled=False),
        localized_deviation=LocalizedDeviationQCConfig(enabled=True, order=3, max_residual_fraction=0.05),
    )


def test_localized_deviation_qc_accepts_smooth_contour():
    """A low-order contour passes with a small residual score."""
    detection = edge([10.0] * 32)
    edges = VesicleEdges(
        extraction_config=EdgeExtractionConfig(),
        detections=[detection],
    )

    edges.run_qc(qc_config())

    assert 'localized_deviation' not in detection.qc.flags
    assert detection.qc.diagnostics["localized_deviation_score"] == pytest.approx(0.0)


def test_localized_deviation_qc_rejects_localized_deviation():
    """A localized contour lobe fails the enabled QC path."""
    radii = np.full(32, 10.0)
    radii[0] = 20.0
    detection = edge(radii)
    edges = VesicleEdges(
        extraction_config=EdgeExtractionConfig(),
        detections=[detection],
    )

    with pytest.raises(ValueError, match="no frames passed"):
        edges.run_qc(qc_config())

    assert 'localized_deviation' in detection.qc.flags
    assert detection.qc.diagnostics.get("localized_deviation_score") > 0.05


def test_support_aware_localized_qc_rejects_narrow_moderate_lobe():
    """A narrow lobe can fail the configured secondary support criterion."""
    radii = np.full(32, 10.0)
    radii[0] = 10.4
    detection = edge(radii)
    config = VesicleQCConfig(
        curvature=CurvatureQCConfig(0.0, enabled=False),
        area=AreaQCConfig(enabled=False),
        localized_deviation=LocalizedDeviationQCConfig(enabled=True, max_residual_fraction=0.05, support_residual_fraction=0.03, max_support_samples=4),
    )

    with pytest.raises(ValueError, match="no frames passed"):
        VesicleEdges(EdgeExtractionConfig(), [detection]).run_qc(config)

    assert detection.qc.diagnostics.get("localized_deviation_support_samples") <= 4


def test_localized_deviation_primary_threshold_is_inclusive(monkeypatch):
    """The primary threshold rejects a score exactly on its boundary."""
    detection = edge([10.0] * 32)
    monkeypatch.setattr(
        "vesmod.VesEdge.qc.checks.localized_deviation_qc.fit_radial_baseline",
        lambda radii, order: type("Fit", (), {"values": np.full(32, 9.5)})(),
    )

    check_localized_deviation(
        detection,
        order=3,
        max_residual_fraction=0.05,
    )

    assert 'localized_deviation' in detection.qc.flags
