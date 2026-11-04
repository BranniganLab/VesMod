"""Tests for singleton-deviation QC and registry integration."""
import numpy as np

from vesmod.VesEdge.qc.experimental.singleton_qc import SingletonDeviationQCConfig
from vesmod.VesEdge.qc.qc_config import VesicleQCConfig
from vesmod.VesEdge.qc.checks.edge_filtering import CurvatureQCConfig
from vesmod.VesEdge.models import EdgeDetection, ImageContour


def make_edge(radii):
    contour = ImageContour(origin=(0, 0), r=np.asarray(radii))
    return EdgeDetection(full_contour=contour, analysis_contour=contour, frame_index=0)


def test_singleton_qc_rejects_narrow_excursion_and_registry_runs_it():
    radii = np.full(120, 40.0)
    radii[30] = 45.0
    edge = make_edge(radii)
    config = VesicleQCConfig(
        curvature=CurvatureQCConfig(threshold=1, enabled=False),
        singleton_deviation=SingletonDeviationQCConfig(enabled=True),
    )
    from vesmod.VesEdge.qc.qc_runner import run_configured_qc_checks
    result = run_configured_qc_checks([edge], config)
    assert 'singleton_deviation' in result.rejected_by(0)
    assert result.diagnostics_for_frame(0)["singleton_deviation"]["singleton_count"] == 1


def test_singleton_qc_preserves_broad_shape():
    radii = np.full(120, 40.0)
    radii[25:35] = 45.0
    edge = make_edge(radii)
    config = VesicleQCConfig(
        curvature=CurvatureQCConfig(threshold=1, enabled=False),
        singleton_deviation=SingletonDeviationQCConfig(enabled=True),
    )
    from vesmod.VesEdge.qc.qc_runner import run_configured_qc_checks
    result = run_configured_qc_checks([edge], config)
    assert 'singleton_deviation' not in result.rejected_by(0)
