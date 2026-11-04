"""Tests for composable VesEdge QC-check orchestration."""

from dataclasses import dataclass
import csv
import sys
from types import SimpleNamespace

import numpy as np


from vesmod.VesEdge.qc import qc_checks, qc_registry
from vesmod.VesEdge.extraction.config import EdgeExtractionConfig
from vesmod.VesEdge.models import EdgeDetection, ImageContour
from vesmod.VesEdge.qc.contract import QCCheck
from vesmod.VesEdge.qc.qc_replay import load_recorded_qc, replay_recorded_qc
from vesmod.VesEdge.vesicle_edges import VesicleEdges
from vesmod.cli import vesedge_cli


def test_builtin_qc_check_order_is_explicit():
    assert [check.name for check in qc_registry.QC_CHECKS] == [
        "curvature",
        "minimum_radius",
        "localized_deviation",
        "singleton_deviation",
        "area",
    ]


def test_registry_runs_enabled_checks_in_declared_order(monkeypatch):
    calls = []

    class Check:
        requires_source_video_frames = False

        def __init__(self, name):
            self.name = name

        def enabled(self, config):
            return True

        def trajectory_flags(self, result):
            return frozenset()

        def run(self, detections, config, frames):
            calls.append(self.name)
            return qc_checks.QCCheckOutcome()

    monkeypatch.setattr(qc_registry, "QC_CHECKS", (Check("first"), Check("second")))

    qc_checks.run_configured_qc_checks([], SimpleNamespace(get_config=lambda name: None))

    assert calls == ["first", "second"]


def test_one_registration_drives_cli_reports_and_replay(tmp_path, monkeypatch):
    """A check with its own settings and diagnostics needs one registry entry."""

    @dataclass(frozen=True)
    class ExampleConfig:
        enabled: bool = False

    def write_diagnostic(path, edges):
        rejected = edges.qc_result.get_result("example")
        path.write_text(f"rejected\n{rejected}\n", encoding="utf-8")

    class ExampleCheck(QCCheck):
        name = "example"
        config_type = ExampleConfig
        requires_source_video_frames = True
        summary_defaults = {"example_rejected": 0}
        diagnostic_writers = ((".example.csv", write_diagnostic),)

        def default_config(self):
            return ExampleConfig()

        def config_to_dict(self, config):
            return {"enabled": config.enabled}

        def enabled(self, config):
            return config.enabled

        def add_cli_arguments(self, parser):
            parser.add_argument("--example-qc", action="store_true")

        def config_from_args(self, args):
            return ExampleConfig(enabled=args.example_qc)

        def run(self, detections, config, frames):
            assert frames is not None
            assert frames[0].shape == (4, 4)
            detections[0].qc.flags.add(self.name)
            return 1

        def summary_values(self, edges):
            return {"example_rejected": edges.qc_result.get_result(self.name)}

    monkeypatch.setattr(qc_registry, "QC_CHECKS", (*qc_registry.QC_CHECKS, ExampleCheck()))
    checkpoint = tmp_path / "sample.npz"
    source = tmp_path / "source.npy"
    frames = np.zeros((2, 4, 4))
    np.save(source, frames)
    contour = ImageContour((0.0, 0.0), np.full(8, 3.0))
    VesicleEdges(
        EdgeExtractionConfig(n_angular_samples=8),
        [EdgeDetection(contour, contour), EdgeDetection(contour, contour)],
        source_path=source,
    ).save_checkpoint(checkpoint)
    output = tmp_path / "qc"
    monkeypatch.setattr(sys, "argv", [
        "vesedge", "qc", str(checkpoint), "--output-dir", str(output), "--example-qc",
    ])

    assert vesedge_cli.main() == 0
    with (output / "qc_summary.csv").open(newline="", encoding="utf-8") as summary:
        assert next(csv.DictReader(summary))["example_rejected"] == "1"
    assert (output / "sample.example.csv").read_text(encoding="utf-8") == "rejected\n1\n"

    selection = load_recorded_qc(output, [checkpoint])
    assert selection.config.get_config("example") == ExampleConfig(enabled=True)
    assert selection.requires_source_video_frames
    edges = VesicleEdges.from_checkpoint(checkpoint)
    assert replay_recorded_qc(edges, checkpoint, selection, frames=frames).accepted_count == 1
    assert edges.successful_detections[0].qc.flags == {"example"}
