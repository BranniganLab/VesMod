import numpy as np

from vesmod.VesEdge import EdgeDetection, ImageContour
from vesmod.VesEdge.edge_filtering import QCFlag
from vesmod.VesEdge.radius_qc import check_radius


def edge(radii):
    radii = np.asarray(radii, dtype=float)
    return EdgeDetection(ImageContour((10.0, 10.0), radii.copy()), ImageContour((10.0, 10.0), radii.copy()))


def test_radius_qc_rejects_collapsed_contour():
    detection = edge([1.0, 2.0, 3.0, 4.0])
    check_radius(detection, 5.0)
    assert detection.qc.median_radius_pixels == 2.5
    assert QCFlag.RADIUS in detection.qc.flags


def test_radius_qc_accepts_threshold_boundary():
    detection = edge([5.0] * 8)
    check_radius(detection, 5.0)
    assert QCFlag.RADIUS not in detection.qc.flags


def test_radius_qc_is_disabled_by_orchestration_config():
    detection = edge([1.0] * 8)
    assert QCFlag.RADIUS not in detection.qc.flags
