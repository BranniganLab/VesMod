"""Unit tests for localized-deviation frame QC."""

import numpy as np
import pytest

from vesmod.VesEdge import (
    AreaQCConfig,
    CurvatureQCConfig,
    EdgeDetection,
    EdgeExtractionConfig,
    ImageContour,
    LocalizedDeviationQCConfig,
    QCFlag,
    VesicleEdges,
    VesicleQCConfig,
    check_localized_deviation,
)


def edge(radii):
    """Build a detection with identical native and analysis contours."""
    contour = ImageContour((10.0, 10.0), np.asarray(radii, dtype=float))
    return EdgeDetection(contour, contour)


def qc_config():
    """Return a config with only localized-deviation QC enabled."""
    return VesicleQCConfig(
        curvature=CurvatureQCConfig(0.0, enabled=False),
        area=AreaQCConfig(enabled=False),
        baseline=LocalizedDeviationQCConfig(
            enabled=True,
            order=3,
            max_residual_fraction=0.05,
        ),
    )


def test_localized_deviation_qc_accepts_smooth_contour():
    """A low-order contour passes with a small residual score."""
    detection = edge([10.0] * 32)
    edges = VesicleEdges(
        extraction_config=EdgeExtractionConfig(),
        detections=[detection],
    )

    edges.run_qc(qc_config())

    assert QCFlag.LOCALIZED_DEVIATION not in detection.qc.flags
    assert detection.qc.localized_deviation_score == pytest.approx(0.0)


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

    assert QCFlag.LOCALIZED_DEVIATION in detection.qc.flags
    assert detection.qc.localized_deviation_score > 0.05


def test_support_aware_localized_qc_rejects_narrow_moderate_lobe():
    """A narrow lobe can fail the configured secondary support criterion."""
    radii = np.full(32, 10.0)
    radii[0] = 10.4
    detection = edge(radii)
    config = VesicleQCConfig(
        curvature=CurvatureQCConfig(0.0, enabled=False),
        area=AreaQCConfig(enabled=False),
        baseline=LocalizedDeviationQCConfig(
            enabled=True,
            max_residual_fraction=0.05,
            support_residual_fraction=0.03,
            max_support_samples=4,
        ),
    )

    with pytest.raises(ValueError, match="no frames passed"):
        VesicleEdges(EdgeExtractionConfig(), [detection]).run_qc(config)

    assert detection.qc.localized_deviation_support_samples <= 4


def test_localized_deviation_primary_threshold_is_inclusive(monkeypatch):
    """The primary threshold rejects a score exactly on its boundary."""
    detection = edge([10.0] * 32)
    monkeypatch.setattr(
        "vesmod.VesEdge.localized_deviation_qc.fit_radial_baseline",
        lambda radii, order: type("Fit", (), {"values": np.full(32, 9.5)})(),
    )

    check_localized_deviation(
        detection,
        order=3,
        max_residual_fraction=0.05,
    )

    assert QCFlag.LOCALIZED_DEVIATION in detection.qc.flags
