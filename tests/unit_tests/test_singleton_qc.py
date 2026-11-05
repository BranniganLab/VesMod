"""Tests for singleton-deviation QC and registry integration."""
import numpy as np
import pytest

from vesmod.VesEdge.qc.experimental.singleton_qc import (
    SingletonDeviationQCConfig,
    check_singleton_deviation,
)
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


@pytest.mark.parametrize("indices, expected_count", [
    ([], 0),
    ([5], 1),
    ([5, 6], 1),
    ([5, 6, 7], 0),
    ([15, 0], 1),
    ([15, 0, 1], 0),
    ([1, 2, 6, 10, 11], 3),
    ([15, 0, 6], 2),
])
def test_circular_excursions_are_counted_once_at_every_rotation(indices, expected_count):
    radii = np.full(16, 40.0)
    radii[indices] = 45.0
    for shift in range(radii.size):
        result = check_singleton_deviation(
            [make_edge(np.roll(radii, shift))], order=0,
            min_residual_fraction=0.05, max_width_samples=2,
        )
        assert result.diagnostics["singleton_count"] == (expected_count,), shift
        assert result.rejected == (expected_count > 0,), shift


@pytest.mark.parametrize("max_width, expected_count", [(15, 0), (16, 1), (17, 1)])
def test_all_above_samples_form_one_full_circular_excursion(max_width, expected_count):
    radii = np.tile([35.0, 45.0], 8)
    result = check_singleton_deviation(
        [make_edge(radii)], order=0,
        min_residual_fraction=0.125, max_width_samples=max_width,
    )
    assert result.diagnostics["singleton_count"] == (expected_count,)
    assert result.rejected == (expected_count > 0,)
