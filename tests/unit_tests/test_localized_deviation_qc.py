"""Unit tests for localized-deviation frame QC."""

import numpy as np
import pytest

from vesmod.VesEdge import (
    AreaQCConfig,
    CurvatureQCConfig,
    EdgeDetection,
    EdgeExtractionConfig,
    EdgeQCConfig,
    ImageContour,
    LocalizedDeviationQCConfig,
    QCFlag,
    VesicleEdges,
)


def edge(radii):
    """Build a detection with identical native and analysis contours."""
    contour = ImageContour((10.0, 10.0), np.asarray(radii, dtype=float))
    return EdgeDetection(contour, contour)


def qc_config():
    """Return a config with only localized-deviation QC enabled."""
    return EdgeQCConfig(
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
