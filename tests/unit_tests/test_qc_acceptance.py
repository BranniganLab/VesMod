"""Acceptance rules combine independent check results without changing them."""

from dataclasses import replace
import csv
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vesmod.VesEdge import AnyFailed, AllFailed
from vesmod.VesEdge.extraction.config import EdgeExtractionConfig
from vesmod.VesEdge.models import EdgeDetection, EdgeDetectionFailure, ImageContour, VesicleQCResult
from vesmod.VesEdge.qc import qc_registry
from vesmod.VesEdge.qc.checks.localized_deviation_qc import LocalizedDeviationQCConfig
from vesmod.VesEdge.qc.experimental.area_qc import AreaQCConfig
from vesmod.VesEdge.qc.qc_config import VesicleQCConfig
from vesmod.VesEdge.qc.qc_replay import load_recorded_qc, replay_recorded_qc
from vesmod.VesEdge.vesicle_edges import VesicleEdges
from vesmod.VesEdge.vesicle_video import VesicleVideo
from vesmod.VesEdge.visualization.animation import draw_vesicle_frame
from vesmod.VesEdge.visualization.trace_plotting import select_trace_detections
from vesmod.cli import vesedge_cli


def make_edges():
    detections = []
    # Both pass, curvature only, area only, both fail, two reference contours.
    for radius, spike in ((10, False), (10, True), (20, False), (20, True), (10, False), (10, False)):
        radii = np.full(120, float(radius))
        if spike:
            radii[0] *= 1.1
        contour = ImageContour((25.0, 25.0), radii)
        detections.append(EdgeDetection(contour, contour))
    detections.insert(1, EdgeDetectionFailure("missing contour"))
    return VesicleEdges(EdgeExtractionConfig(), detections)


def config(rule=None):
    return VesicleQCConfig(
        area=AreaQCConfig(enabled=True),
        localized_deviation=LocalizedDeviationQCConfig(enabled=False),
        reject_when=rule,
    )


@pytest.mark.parametrize("rule, expected", [
    (None, (0, 5, 6)),
    (AnyFailed("curvature", "area"), (0, 5, 6)),
    (AllFailed("curvature", "area"), (0, 2, 3, 5, 6)),
    (AnyFailed("curvature"), (0, 3, 5, 6)),
    (AllFailed("area"), (0, 2, 5, 6)),
])
def test_rules_cover_all_failure_combinations_with_source_frame_gaps(rule, expected):
    edges = make_edges()
    edges.run_qc(config(rule))
    recorded = edges.qc_result
    assert recorded.frame_indices == (0, 2, 3, 4, 5, 6)
    assert recorded.rejected_by(0) == frozenset()
    assert recorded.rejected_by(2) == {"curvature"}
    assert recorded.rejected_by(3) == {"area"}
    assert recorded.rejected_by(4) == {"curvature", "area"}
    assert recorded.accepted_frame_indices == expected
    assert tuple(edge.frame_index for edge in edges.accepted_detections) == expected
    restored = VesicleQCResult.from_dict(json.loads(json.dumps(recorded.to_dict(), allow_nan=False)))
    assert restored.accepted_frame_indices == expected
    assert restored.config.reject_when == rule
    assert all(not hasattr(edge, "qc") for edge in edges.successful_detections)
    with pytest.raises(KeyError, match="not evaluated"):
        recorded.diagnostics_for_frame(1)


def test_rule_does_not_change_measurements_and_check_order_does_not_change_area(monkeypatch):
    edges = make_edges()
    edges.run_qc(config(AnyFailed("curvature", "area")))
    original = {name: result.to_dict() for name, result in edges.qc_result.results.items()}
    monkeypatch.setattr(qc_registry, "QC_CHECKS", tuple(reversed(qc_registry.QC_CHECKS)))
    edges.run_qc(config(AllFailed("curvature", "area")))
    assert {name: result.to_dict() for name, result in edges.qc_result.results.items()} == original
    assert edges.qc_result.accepted_frame_indices == (0, 2, 3, 5, 6)


@pytest.mark.parametrize("rule, message", [
    (AnyFailed("unknown"), "unknown"),
    (AllFailed("area"), "disabled"),
])
def test_rule_cannot_reference_an_unknown_or_disabled_check(rule, message):
    with pytest.raises(ValueError, match=message):
        VesicleQCConfig(reject_when=rule)


@pytest.mark.parametrize("names", [(), ("curvature", "curvature"), ("",), (AllFailed("curvature"),)])
def test_rules_require_distinct_check_names_and_do_not_accept_nested_rules(names):
    with pytest.raises(ValueError):
        AnyFailed(*names)


def test_all_disabled_checks_accept_every_successful_frame():
    edges = make_edges()
    edges.run_qc(VesicleQCConfig(checks={
        check.name: replace(check.default_config(), enabled=False)
        for check in qc_registry.QC_CHECKS
    }))
    assert edges.qc_result.accepted_frame_indices == (0, 2, 3, 4, 5, 6)


def test_recording_rejection_is_a_veto_even_when_frame_rule_accepts():
    edges = make_edges()
    edges.run_qc(config(AllFailed("curvature", "area")))
    recorded = edges.qc_result
    result = recorded.get_result("curvature")
    results = dict(recorded.results)
    results["curvature"] = replace(result, trajectory_flags=frozenset({"unusable_recording"}))
    vetoed = replace(recorded, results=results)
    assert not vetoed.passed
    assert vetoed.accepted_frame_indices == ()


def test_rendering_and_trace_selection_follow_the_rule():
    edges = make_edges()
    edges.run_qc(config(AllFailed("curvature", "area")))
    assert select_trace_detections(edges, [2])[0].frame_index == 2
    figure, axis = plt.subplots()
    try:
        video = VesicleVideo(np.zeros((7, 50, 50)))
        draw_vesicle_frame(video, axis, 2, edges=edges)
        assert axis.lines[0].get_color() == "tab:green"
        draw_vesicle_frame(video, axis, 4, edges=edges)
        assert axis.lines[0].get_color() == "tab:red"
    finally:
        plt.close(figure)


@pytest.mark.parametrize("option, expected_count", [
    ("--reject-if-any-fail", 3), ("--reject-if-all-fail", 5),
])
def test_cli_saves_the_rule_and_downstream_replay_preserves_selection(tmp_path, monkeypatch, option, expected_count):
    checkpoint = tmp_path / "sample.npz"
    edges = make_edges()
    edges.save_checkpoint(checkpoint)
    output = tmp_path / "qc"
    monkeypatch.setattr(sys, "argv", [
        "vesedge", "qc", str(checkpoint), "--area-qc", "--no-localized-deviation-qc",
        option, "curvature", "area", "--output-dir", str(output),
    ])
    assert vesedge_cli.main() == 0
    assert np.load(output / "sample.npy").shape[0] == expected_count
    with (output / "qc_summary.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert int(row["accepted"]) == expected_count
    assert int(row["curvature_rejected"]) == 2
    assert int(row["area_rejected"]) == 2
    selection = load_recorded_qc(output, [checkpoint])
    restored = VesicleEdges.from_checkpoint(checkpoint)
    replay = replay_recorded_qc(restored, checkpoint, selection)
    assert replay.accepted_count == expected_count
