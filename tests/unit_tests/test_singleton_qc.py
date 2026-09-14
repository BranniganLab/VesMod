"""Tests for singleton-deviation QC and registry integration."""
import numpy as np

from vesmod.VesEdge import SingletonDeviationQCConfig, VesicleQCConfig
from vesmod.VesEdge.config import CurvatureQCConfig
from vesmod.VesEdge.models import EdgeQC, EdgeDetection, ImageContour, QCFlag


def make_edge(radii):
    contour = ImageContour(origin=(0, 0), r=np.asarray(radii))
    return EdgeDetection(full_contour=contour, analysis_contour=contour, qc=EdgeQC())


def test_singleton_qc_rejects_narrow_excursion_and_registry_runs_it():
    radii = np.full(120, 40.0)
    radii[30] = 45.0
    edge = make_edge(radii)
    config = VesicleQCConfig(curvature=CurvatureQCConfig(threshold=1, enabled=False), singleton=SingletonDeviationQCConfig(enabled=True))
    from vesmod.VesEdge.qc_checks import run_configured_qc_checks
    run_configured_qc_checks([edge], config)
    assert QCFlag.SINGLETON_DEVIATION in edge.qc.flags
    assert edge.qc.singleton_count == 1


def test_singleton_qc_preserves_broad_shape():
    radii = np.full(120, 40.0)
    radii[25:35] = 45.0
    edge = make_edge(radii)
    config = VesicleQCConfig(curvature=CurvatureQCConfig(threshold=1, enabled=False), singleton=SingletonDeviationQCConfig(enabled=True))
    from vesmod.VesEdge.qc_checks import run_configured_qc_checks
    run_configured_qc_checks([edge], config)
    assert QCFlag.SINGLETON_DEVIATION not in edge.qc.flags
