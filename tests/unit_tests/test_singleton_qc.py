"""Tests for singleton-deviation QC and registry integration."""
import numpy as np

from vesmod.VesEdge.experimental.singleton_qc import SingletonDeviationQCConfig
from vesmod.VesEdge.qc.qc_config import VesicleQCConfig
from vesmod.VesEdge.qc.checks.edge_filtering import CurvatureQCConfig
from vesmod.VesEdge.models import EdgeQC, EdgeDetection, ImageContour


def make_edge(radii):
    contour = ImageContour(origin=(0, 0), r=np.asarray(radii))
    return EdgeDetection(full_contour=contour, analysis_contour=contour, qc=EdgeQC())


def test_singleton_qc_rejects_narrow_excursion_and_registry_runs_it():
    radii = np.full(120, 40.0)
    radii[30] = 45.0
    edge = make_edge(radii)
    config = VesicleQCConfig(
        curvature=CurvatureQCConfig(threshold=1, enabled=False),
        singleton_deviation=SingletonDeviationQCConfig(enabled=True),
    )
    from vesmod.VesEdge.qc.qc_checks import run_configured_qc_checks
    run_configured_qc_checks([edge], config)
    assert 'singleton_deviation' in edge.qc.flags
    assert edge.qc.diagnostics["singleton_count"] == 1


def test_singleton_qc_preserves_broad_shape():
    radii = np.full(120, 40.0)
    radii[25:35] = 45.0
    edge = make_edge(radii)
    config = VesicleQCConfig(
        curvature=CurvatureQCConfig(threshold=1, enabled=False),
        singleton_deviation=SingletonDeviationQCConfig(enabled=True),
    )
    from vesmod.VesEdge.qc.qc_checks import run_configured_qc_checks
    run_configured_qc_checks([edge], config)
    assert 'singleton_deviation' not in edge.qc.flags
