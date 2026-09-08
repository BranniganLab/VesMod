"""Tests for composable VesEdge QC-check orchestration."""

from types import SimpleNamespace

import pytest

from vesmod.VesEdge import EdgeQCConfig
from vesmod.VesEdge.config import (
    AreaQCConfig,
    CurvatureQCConfig,
    InternalVesicleQCConfig,
)
from vesmod.VesEdge import qc_checks


def test_builtin_qc_check_order_is_explicit():
    assert [check.name for check in qc_checks.QC_CHECKS] == [
        "curvature",
        "minimum_radius",
        "localized_deviation",
        "singleton_deviation",
        "area",
        "internal_vesicle",
    ]


def test_registry_runs_enabled_checks_in_declared_order(monkeypatch):
    calls = []

    class Check:
        requires_frames = False

        def __init__(self, name):
            self.name = name

        def enabled(self, config):
            return True

        def run(self, detections, config, frames):
            calls.append(self.name)
            return qc_checks.QCCheckOutcome()

    monkeypatch.setattr(qc_checks, "QC_CHECKS", (Check("first"), Check("second")))

    qc_checks.run_configured_qc_checks([], SimpleNamespace())

    assert calls == ["first", "second"]


def test_registry_requires_frames_only_for_enabled_frame_check():
    config = EdgeQCConfig(
        curvature=CurvatureQCConfig(threshold=0.1, enabled=False),
        area=AreaQCConfig(max_relative_deviation=0.25, enabled=False),
        internal_vesicle=InternalVesicleQCConfig(enabled=True),
    )

    with pytest.raises(ValueError, match="Internal-vesicle QC is enabled"):
        qc_checks.run_configured_qc_checks([], config)
