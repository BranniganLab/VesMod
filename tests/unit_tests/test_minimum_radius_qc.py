import numpy as np
import pytest

from vesmod.VesEdge import (
    CurvatureQCConfig,
    EdgeDetection,
    EdgeExtractionConfig,
    EdgeQCConfig,
    ImageContour,
    MinimumRadiusQCConfig,
    VesicleEdges,
)
from vesmod.VesEdge.edge_filtering import QCFlag
from vesmod.VesEdge.minimum_radius_qc import check_minimum_radius


def edge(radii):
    radii = np.asarray(radii, dtype=float)
    return EdgeDetection(ImageContour((10.0, 10.0), radii.copy()), ImageContour((10.0, 10.0), radii.copy()))


def test_minimum_radius_qc_rejects_collapsed_contour():
    detection = edge([1.0, 2.0, 3.0, 4.0])
    check_minimum_radius(detection, 5.0)
    assert detection.qc.median_radius_pixels == 2.5
    assert QCFlag.MINIMUM_RADIUS in detection.qc.flags


def test_minimum_radius_qc_accepts_threshold_boundary():
    detection = edge([5.0] * 8)
    check_minimum_radius(detection, 5.0)
    assert QCFlag.MINIMUM_RADIUS not in detection.qc.flags


def test_minimum_radius_qc_is_disabled_by_orchestration_config():
    detection = edge([1.0] * 8)
    edges = VesicleEdges(EdgeExtractionConfig(), [detection])
    config = EdgeQCConfig(
        curvature=CurvatureQCConfig(threshold=5.0),
        minimum_radius=MinimumRadiusQCConfig(enabled=False),
    )

    edges.run_qc(config)

    assert QCFlag.MINIMUM_RADIUS not in detection.qc.flags


@pytest.mark.parametrize(
    "invalid_radius",
    [0.0, -1.0, np.nan, np.inf, -np.inf],
)
def test_minimum_radius_qc_rejects_invalid_radii(invalid_radius):
    detection = edge([5.0] * 8)
    object.__setattr__(
        detection.full_contour,
        "r",
        np.full(8, invalid_radius),
    )

    check_minimum_radius(detection, 5.0)

    assert QCFlag.MINIMUM_RADIUS in detection.qc.flags
