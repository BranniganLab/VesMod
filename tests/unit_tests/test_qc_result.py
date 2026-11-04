"""Validation and persistence of standard QC decisions and declared measurements."""

from dataclasses import dataclass, replace
import json

import numpy as np
import pytest

from vesmod.VesEdge.models import EdgeDetection, ImageContour, VesicleQCResult
from vesmod.VesEdge.extraction.config import EdgeExtractionConfig
from vesmod.VesEdge.vesicle_edges import VesicleEdges
from vesmod.VesEdge.qc import qc_registry
from vesmod.VesEdge.qc.qc_check import QCCheck
from vesmod.VesEdge.qc.qc_config import VesicleQCConfig
from vesmod.VesEdge.qc.qc_registry import QC_CHECKS
from vesmod.VesEdge.qc.qc_result import QCResult


@dataclass(frozen=True)
class Settings:
    enabled: bool = True


class ScoreCheck(QCCheck):
    name = "score"
    config_type = Settings
    diagnostic_fields = ("score",)
    extra_result_fields = ("reference",)

    def run(self, detections, config, frames):
        raise NotImplementedError


def result(**overrides):
    values = {
        "check_name": "score",
        "frame_indices": (2, 8),
        "rejected": (False, True),
        "diagnostics": {"score": (0.1, 0.8)},
        "extras": {"reference": 0.5},
    }
    return QCResult(**(values | overrides))


@pytest.mark.parametrize("overrides", [
    {"frame_indices": (2, 2)},
    {"frame_indices": (8, 2)},
    {"frame_indices": (-1, 2)},
    {"frame_indices": (True, 2)},
    {"rejected": (False,)},
    {"rejected": (0, 1)},
    {"diagnostics": {"score": (0.1,)}},
    {"trajectory_flags": frozenset({""})},
])
def test_invalid_standard_fields_are_rejected(overrides):
    with pytest.raises(ValueError):
        result(**overrides)


def test_measurements_have_a_strict_json_round_trip():
    measured = result(
        diagnostics={"score": (np.float64(0.1), float("nan"))},
        extras={"reference": {"value": np.int64(3), "missing": float("inf")}},
    )
    data = json.loads(json.dumps(measured.to_dict(), allow_nan=False))
    restored = QCResult.from_dict(data)
    assert restored == measured
    assert restored.diagnostics["score"] == (0.1, None)
    assert restored.extras["reference"] == {"value": 3, "missing": None}
    assert restored.rejected_count == 1


def test_arbitrary_objects_cannot_be_hidden_in_extra_measurements():
    with pytest.raises(TypeError, match="Unsupported QC measurement"):
        result(extras={"reference": object()})


@pytest.mark.parametrize("value, error", [
    (None, TypeError),
    (result(check_name="other"), ValueError),
    (result(diagnostics={"undeclared": (0.1, 0.8)}), ValueError),
    (result(extras={"undeclared": 3}), ValueError),
    (result(extras={}), ValueError),
])
def test_check_validates_standard_result_and_declared_fields(value, error):
    with pytest.raises(error):
        ScoreCheck().validate_result(value)


def test_declared_result_fields_are_loaded_through_the_check():
    check = ScoreCheck()
    assert check.result_from_dict(result().to_dict()) == result()


def test_all_disabled_run_still_records_its_accepted_frames():
    config = VesicleQCConfig(checks={
        check.name: replace(check.default_config(), enabled=False)
        for check in QC_CHECKS
    })
    recorded = VesicleQCResult(config=config, frame_indices=(2, 8))
    restored = VesicleQCResult.from_dict(
        json.loads(json.dumps(recorded.to_dict(), allow_nan=False))
    )
    assert restored.accepted_frame_indices == (2, 8)


def test_recording_rejection_does_not_need_to_reject_individual_frames():
    recorded = VesicleQCResult(
        config=VesicleQCConfig(),
        frame_indices=(2, 8),
        results={"score": result(rejected=(False, False), trajectory_flags=frozenset({"unstable"}))},
    )
    assert recorded.accepted_frame_indices == ()
    assert recorded.get_result("score").rejected_count == 0



@pytest.mark.parametrize("malformed", ["none", "missing_frame", "undeclared_extra"])
def test_invalid_check_result_restores_the_previous_completed_run(monkeypatch, malformed):
    contour = ImageContour((0.0, 0.0), np.full(120, 10.0))
    edges = VesicleEdges(
        EdgeExtractionConfig(),
        [EdgeDetection(contour, contour), EdgeDetection(contour, contour)],
    )
    edges.run_qc(VesicleQCConfig())
    previous = edges.qc_result

    class InvalidCheck(ScoreCheck):
        def run(self, detections, config, frames):
            if malformed == "none":
                return None
            indices = (0,) if malformed == "missing_frame" else (0, 1)
            return QCResult(
                check_name=self.name,
                frame_indices=indices,
                rejected=tuple(False for _ in indices),
                diagnostics={"score": tuple(0.0 for _ in indices)},
                extras={"wrong": 3} if malformed == "undeclared_extra" else {"reference": 3},
            )

    monkeypatch.setattr(qc_registry, "QC_CHECKS", (InvalidCheck(),))
    with pytest.raises((TypeError, ValueError)):
        edges.run_qc(VesicleQCConfig())
    assert edges.qc_result is previous
    assert all(not hasattr(edge, "qc") for edge in edges.successful_detections)


def test_runner_retains_a_completed_recording_rejection(monkeypatch):
    class RecordingCheck(ScoreCheck):
        def run(self, detections, config, frames):
            return QCResult(
                check_name=self.name,
                frame_indices=tuple(edge.frame_index for edge in detections),
                rejected=(False,),
                diagnostics={"score": (0.0,)},
                trajectory_flags=frozenset({"unstable"}),
                extras={"reference": 0.0},
            )

    monkeypatch.setattr(qc_registry, "QC_CHECKS", (RecordingCheck(),))
    contour = ImageContour((0.0, 0.0), np.full(120, 10.0))
    edges = VesicleEdges(EdgeExtractionConfig(), [EdgeDetection(contour, contour)])
    with pytest.raises(ValueError):
        edges.run_qc(VesicleQCConfig())
    assert edges.qc_result.trajectory_flags == frozenset({"unstable"})
    assert edges.qc_result.accepted_frame_indices == ()
    assert not edges.qc_result.rejected_by(0)
