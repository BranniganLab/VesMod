import numpy as np
import pytest

from vesmod.VesEdge.qc.checks.edge_filtering import CurvatureQCConfig
from vesmod.VesEdge.models import EdgeDetection, ImageContour
from vesmod.VesEdge.extraction.config import EdgeExtractionConfig
from vesmod.VesEdge.qc.experimental.minimum_radius_qc import MinimumRadiusQCConfig
from vesmod.VesEdge.vesicle_edges import VesicleEdges
from vesmod.VesEdge.qc.qc_config import VesicleQCConfig
from vesmod.VesEdge.qc.experimental.minimum_radius_qc import check_minimum_radius


def edge(radii):
    radii = np.asarray(radii, dtype=float)
    return EdgeDetection(ImageContour((10.0, 10.0), radii.copy()), ImageContour((10.0, 10.0), radii.copy()), frame_index=0)


def test_minimum_radius_qc_rejects_collapsed_contour():
    detection = edge([1.0, 2.0, 3.0, 4.0])
    result = check_minimum_radius([detection], 5.0)
    assert result.diagnostics["median_radius_pixels"][0] == 2.5
    assert result.rejected[0]


def test_minimum_radius_qc_accepts_threshold_boundary():
    detection = edge([5.0] * 8)
    result = check_minimum_radius([detection], 5.0)
    assert not result.rejected[0]


def test_minimum_radius_qc_is_disabled_by_orchestration_config():
    detection = edge([1.0] * 8)
    edges = VesicleEdges(EdgeExtractionConfig(), [detection])
    config = VesicleQCConfig(
        curvature=CurvatureQCConfig(threshold=5.0),
        minimum_radius=MinimumRadiusQCConfig(enabled=False),
    )

    edges.run_qc(config)

    assert edges.qc_result.get_result("minimum_radius") is None


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

    result = check_minimum_radius([detection], 5.0)

    assert result.rejected[0]
