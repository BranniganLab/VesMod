"""Tests for the VesEdge QC command and its main CLI dispatch."""

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from vesmod.VesEdge.experimental.area_qc import AreaQCConfig, AreaQCResult
from vesmod.VesEdge.qc.checks.edge_filtering import CurvatureQCConfig
from vesmod.VesEdge.qc.checks.localized_deviation_qc import LocalizedDeviationQCConfig
from vesmod.VesEdge.models import EdgeDetection, ImageContour, QCFlag, VesicleQCResult
from vesmod.VesEdge.experimental.area_qc import _save_area_qc_plot, _write_area_qc_csv
from vesmod.VesEdge.experimental.minimum_radius_qc import _write_minimum_radius_qc_csv
from vesmod.VesEdge.extraction.config import EdgeExtractionConfig
from vesmod.VesEdge.vesicle_edges import VesicleEdges
from vesmod.VesEdge.qc.qc_config import VesicleQCConfig
from vesmod.VesEdge.cli import qc_cli
from vesmod.cli import vesedge_cli


def _qc_args(tmp_path, input_path=Path("sample.npz")) -> argparse.Namespace:
    """Return standard QC CLI arguments."""
    return argparse.Namespace(
        input_path=input_path,
        output_dir=tmp_path,
        recursive=False,
        overwrite=True,
        curvature_threshold=5.0,
        no_curvature_qc=False,
        max_relative_area_deviation=0.25,
        area_qc=False,
        singleton_deviation_qc=False,
        singleton_deviation_order=3,
        min_singleton_deviation_residual_fraction=0.05,
        max_singleton_deviation_width_samples=2,
    )


def test_parse_args_selects_qc_subcommand(monkeypatch, tmp_path):
    """Test QC arguments are scoped to the qc subcommand."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "qc",
            "checkpoints",
            "--output-dir",
            str(tmp_path),
            "--no-curvature-qc",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.command == "qc"
    assert args.input_path == Path("checkpoints")
    assert args.output_dir == tmp_path
    assert args.no_curvature_qc
    assert args.curvature_threshold == pytest.approx(21.520619405632544)
    assert args.max_relative_area_deviation == pytest.approx(0.25)
    assert not args.area_qc
    assert args.localized_deviation_qc
    config = qc_cli._qc_config_from_args(args)
    assert not config.curvature.enabled
    assert config.localized_deviation.enabled
    assert not config.area.enabled
    assert not config.minimum_radius.enabled
    assert not config.singleton_deviation.enabled


def test_qc_parser_accepts_singleton_deviation_options(monkeypatch, tmp_path):
    """Test singleton-deviation options are exposed by the QC CLI."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "qc",
            "checkpoints",
            "--output-dir",
            str(tmp_path),
            "--singleton-deviation-qc",
            "--singleton-deviation-order",
            "4",
            "--min-singleton-deviation-residual-fraction",
            "0.08",
            "--max-singleton-deviation-width-samples",
            "3",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.singleton_deviation_qc
    assert args.singleton_deviation_order == 4
    assert args.min_singleton_deviation_residual_fraction == pytest.approx(0.08)
    assert args.max_singleton_deviation_width_samples == 3
    config = qc_cli._qc_config_from_args(args)
    assert config.singleton_deviation.enabled
    assert config.singleton_deviation.order == 4
    assert config.singleton_deviation.min_residual_fraction == pytest.approx(0.08)
    assert config.singleton_deviation.max_width_samples == 3


def test_qc_parser_accepts_localized_deviation_support_options(
    monkeypatch,
    tmp_path,
):
    """Test support-aware localized-deviation settings reach its check config."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "qc",
            "checkpoints",
            "--output-dir",
            str(tmp_path),
            "--localized-deviation-qc",
            "--localized-deviation-support-residual-fraction",
            "0.03",
            "--max-localized-deviation-support-samples",
            "4",
        ],
    )

    config = qc_cli._qc_config_from_args(vesedge_cli.parse_args())

    assert config.localized_deviation.support_residual_fraction == pytest.approx(0.03)
    assert config.localized_deviation.max_support_samples == 4


@pytest.mark.parametrize(
    "removed_option",
    [
        "--population-bic-threshold",
        "--max-minor-population-fraction",
        "--no-population-qc",
    ],
)
def test_parse_args_rejects_removed_population_options(
    monkeypatch,
    tmp_path,
    removed_option,
):
    """Test removed population-QC options are not silently ignored."""
    argv = [
        "vesedge",
        "qc",
        "checkpoints",
        "--output-dir",
        str(tmp_path),
        removed_option,
    ]
    if removed_option != "--no-population-qc":
        argv.append("10")
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        vesedge_cli.parse_args()


def test_process_qc_file_runs_qc_and_saves_filtered_output(tmp_path, monkeypatch):
    """Test QC loads a checkpoint, applies QC, and writes accepted contours."""
    observed = {}

    class Detection:
        def __init__(self):
            self.qc = argparse.Namespace(flags=set(), passed=True)

    class FakeEdges:
        def __init__(self):
            self.detections = [Detection(), Detection()]
            self.successful_detections = self.detections
            self.qc_result = None

        def run_qc(self, config):
            observed["qc_config"] = config
            self.qc_result = VesicleQCResult(config)

        def export_accepted_radii(self, path):
            observed["npy"] = path

    path = Path("sample.npz")
    edges = FakeEdges()
    monkeypatch.setattr(
        qc_cli.VesicleEdges,
        "from_checkpoint",
        lambda checkpoint_path: edges,
    )
    args = _qc_args(tmp_path, path)
    config = qc_cli._qc_config_from_args(args)

    row = qc_cli.process_qc_file(path, args, config)

    assert observed["npy"] == tmp_path / "sample.npy"
    assert observed["qc_config"] is config
    assert row["accepted"] == 2
    assert row["status"] == "ok"


def test_process_qc_file_preserves_relative_output_path(tmp_path, monkeypatch):
    """Test recursive QC outputs preserve checkpoint subdirectories."""
    observed = {}

    class Detection:
        def __init__(self):
            self.qc = argparse.Namespace(flags=set(), passed=True)

    class FakeEdges:
        detections = [Detection()]
        successful_detections = detections
        qc_result = None

        def run_qc(self, config):
            pass

        def export_accepted_radii(self, path):
            observed["npy"] = path

    input_root = tmp_path / "checkpoints"
    path = input_root / "nested" / "sample.npz"
    output_dir = tmp_path / "qc"
    monkeypatch.setattr(
        qc_cli.VesicleEdges,
        "from_checkpoint",
        lambda checkpoint_path: FakeEdges(),
    )
    args = _qc_args(output_dir, input_root)
    config = qc_cli._qc_config_from_args(args)

    qc_cli.process_qc_file(path, args, config)

    assert observed["npy"] == output_dir / "nested" / "sample.npy"


def test_process_qc_file_records_zero_accepted_frames(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Test a completed QC run with no accepted frames is summarized."""

    class Detection:
        def __init__(self):
            self.qc = argparse.Namespace(flags=set(), passed=False)

    class FakeEdges:
        def __init__(self):
            self.detections = [Detection()]
            self.successful_detections = self.detections
            self.qc_result = None

        def run_qc(self, config):
            self.qc_result = VesicleQCResult(config)
            raise ValueError("no frames passed quality control")

        def export_accepted_radii(self, path):
            raise AssertionError("No .npy should be saved")

    path = Path("sample.npz")
    edges = FakeEdges()
    monkeypatch.setattr(
        qc_cli.VesicleEdges,
        "from_checkpoint",
        lambda checkpoint_path: edges,
    )
    args = _qc_args(tmp_path, path)
    config = qc_cli._qc_config_from_args(args)
    stale_output = tmp_path / "sample.npy"
    np.save(stale_output, np.ones((2, 12)))

    row = qc_cli.process_qc_file(path, args, config)

    captured = capsys.readouterr()
    assert str(path.resolve()) in captured.out

    assert row["accepted"] == 0
    assert row["status"] == "no_accepted_frames"
    assert "no frames passed quality control" in row["error"]
    assert not stale_output.exists()


def test_process_qc_file_returns_load_error_summary(tmp_path, monkeypatch):
    """Test checkpoint load failures still produce canonical summary rows."""
    path = Path("broken.npz")
    monkeypatch.setattr(
        qc_cli.VesicleEdges,
        "from_checkpoint",
        lambda checkpoint_path: (_ for _ in ()).throw(ValueError("bad checkpoint")),
    )
    args = _qc_args(tmp_path, path)
    config = qc_cli._qc_config_from_args(args)

    row = qc_cli.process_qc_file(path, args, config)

    assert row == {
        "file": "broken.npz",
        "frames": 0,
        "successful_detections": 0,
        "extraction_failures": 0,
        "curvature_rejected": 0,
        "area_rejected": 0,
        "minimum_radius_rejected": 0,
        "singleton_deviation_rejected": 0,
        "localized_deviation_rejected": 0,
        "accepted": 0,
        "accepted_fraction": 0.0,
        "status": "load_error",
        "error": "bad checkpoint",
    }


def test_write_qc_provenance_rejects_different_configuration(tmp_path):
    """Test a QC directory cannot silently mix configurations."""
    checkpoint = tmp_path / "sample.npz"
    config = VesicleQCConfig(curvature=CurvatureQCConfig(threshold=5.0))
    qc_cli._write_qc_provenance(
        tmp_path,
        config,
        checkpoint,
        False,
        [checkpoint],
        overwrite=False,
    )

    different = VesicleQCConfig(curvature=CurvatureQCConfig(threshold=8.0))
    with pytest.raises(ValueError, match="different input selection or QC configuration"):
        qc_cli._write_qc_provenance(
            tmp_path,
            different,
            checkpoint,
            False,
            [checkpoint],
            overwrite=False,
        )


def test_write_qc_provenance_records_manifest_and_recursive_setting(tmp_path):
    """Test provenance identifies the exact resolved checkpoint selection."""
    input_root = tmp_path / "checkpoints"
    first = input_root / "a.npz"
    second = input_root / "nested" / "b.npz"
    config = VesicleQCConfig(curvature=CurvatureQCConfig(threshold=5.0))

    qc_cli._write_qc_provenance(
        tmp_path / "qc",
        config,
        input_root,
        True,
        [first, second],
        overwrite=False,
    )

    data = json.loads((tmp_path / "qc" / "vesedge_qc.json").read_text())
    assert data["recursive"] is True
    assert data["checkpoint_manifest"] == [str(first.resolve()), str(second.resolve())]
    assert data["qc_config"]["curvature"]["threshold"] == 5.0


def test_overwrite_incompatible_provenance_removes_stale_outputs(tmp_path):
    """Test incompatible overwrite clears only recorded QC artifacts."""
    output_dir = tmp_path / "qc"
    checkpoint = tmp_path / "sample.npz"
    config = VesicleQCConfig(curvature=CurvatureQCConfig(threshold=5.0))
    qc_cli._write_qc_provenance(
        output_dir,
        config,
        checkpoint,
        False,
        [checkpoint],
        overwrite=False,
    )
    stale = output_dir / "nested" / "orphan.npy"
    stale.parent.mkdir(parents=True)
    stale.touch()
    unrelated = output_dir / "unrelated.npy"
    unrelated.touch()
    (output_dir / "qc_summary.csv").write_text("old")
    provenance_path = output_dir / "vesedge_qc.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["managed_artifacts"] = ["nested/orphan.npy"]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    different = VesicleQCConfig(curvature=CurvatureQCConfig(threshold=8.0))
    qc_cli._write_qc_provenance(
        output_dir,
        different,
        checkpoint,
        False,
        [checkpoint],
        overwrite=True,
    )

    assert not stale.exists()
    assert unrelated.is_file()
    assert not (output_dir / "qc_summary.csv").exists()
    data = json.loads((output_dir / "vesedge_qc.json").read_text())
    assert data["qc_config"]["curvature"]["threshold"] == 8.0


def test_matching_qc_overwrite_replaces_provenance_after_cleanup(tmp_path):
    """Test a matching overwrite cleans once and writes new provenance."""
    output_dir = tmp_path / "qc"
    checkpoint = tmp_path / "sample.npz"
    config = VesicleQCConfig(curvature=CurvatureQCConfig(threshold=5.0))
    qc_cli._write_qc_provenance(
        output_dir, config, checkpoint, False, [checkpoint], overwrite=False
    )
    managed = output_dir / "sample.npy"
    managed.touch()
    provenance_path = output_dir / "vesedge_qc.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["managed_artifacts"] = ["sample.npy"]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    qc_cli._write_qc_provenance(
        output_dir, config, checkpoint, False, [checkpoint], overwrite=True
    )

    assert not managed.exists()
    assert provenance_path.is_file()
    assert json.loads(provenance_path.read_text())["managed_artifacts"] == []


def test_qc_manifest_excludes_preexisting_expected_outputs(tmp_path):
    """Test preserved outputs are not adopted by the current-run manifest."""
    output_dir = tmp_path / "qc"
    output_dir.mkdir()
    expected = output_dir / "sample.npy"
    expected.touch()
    provenance = {
        "input_path": str(tmp_path.resolve()),
        "recursive": False,
        "checkpoint_manifest": [],
        "qc_config": {},
        "managed_artifacts": [],
    }
    (output_dir / "vesedge_qc.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )

    qc_cli._record_qc_artifacts(output_dir, set())

    recorded = json.loads(
        (output_dir / "vesedge_qc.json").read_text(encoding="utf-8")
    )
    assert recorded["managed_artifacts"] == []
    assert expected.is_file()


def test_write_qc_summary_writes_batch_csv(tmp_path):
    """Test QC summary output contains one row per processed checkpoint."""
    rows = [
        {
            "file": "sample.npz",
            "frames": 10,
            "successful_detections": 9,
            "extraction_failures": 1,
            "curvature_rejected": 2,
            "area_rejected": 1,
            "accepted": 7,
            "accepted_fraction": 7 / 9,
            "status": "ok",
            "error": "",
        }
    ]

    qc_cli._write_qc_summary(tmp_path, rows)

    summary = (tmp_path / "qc_summary.csv").read_text()
    assert "sample.npz" in summary
    assert "curvature_rejected" in summary
    assert "area_rejected" in summary
    assert "singleton_deviation_rejected" in summary
    assert ",7," in summary


@pytest.mark.parametrize(
    "manifest",
    [
        {"managed_artifacts": ["../outside.npy"]},
        {"managed_artifacts": ["nested/data.txt"]},
        {"managed_artifacts": "not-a-list"},
    ],
)
def test_qc_overwrite_rejects_unsafe_or_malformed_manifest(tmp_path, manifest):
    """Test QC refuses cleanup unless its artifact manifest is trustworthy."""
    output_dir = tmp_path / "qc"
    output_dir.mkdir()
    provenance = {
        "checkpoint_manifest": [],
        "qc_config": {},
        **manifest,
    }
    (output_dir / "vesedge_qc.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    unrelated = output_dir / "unrelated.npy"
    unrelated.touch()

    with pytest.raises(ValueError, match="refusing to remove files"):
        qc_cli._remove_managed_qc_artifacts(output_dir)

    assert unrelated.is_file()


def test_run_qc_writes_summary_when_every_checkpoint_fails_to_load(
    tmp_path,
    monkeypatch,
):
    """Test an all-load-failure batch still writes qc_summary.csv."""
    input_dir = tmp_path / "checkpoints"
    input_dir.mkdir()
    first = input_dir / "first.npz"
    second = input_dir / "second.npz"
    first.touch()
    second.touch()
    output_dir = tmp_path / "qc"
    args = _qc_args(output_dir, input_dir)

    monkeypatch.setattr(
        qc_cli.VesicleEdges,
        "from_checkpoint",
        lambda checkpoint_path: (_ for _ in ()).throw(ValueError("bad checkpoint")),
    )

    qc_cli.run(args)

    summary = (output_dir / "qc_summary.csv").read_text()
    assert "first.npz" in summary
    assert "second.npz" in summary
    assert summary.count("load_error") == 2


def test_area_qc_diagnostics_preserve_frame_measurements(tmp_path):
    """Test area QC writes exact values and a trajectory diagnostic plot."""

    class Detection:
        frame_index = 4
        qc = argparse.Namespace(flags={QCFlag.AREA_DEVIATION})

    class FakeEdges:
        successful_detections = [Detection()]
        qc_result = VesicleQCResult(
            results={"area": AreaQCResult(
                areas_pixels2=(25.0,),
                reference_area_pixels2=100.0,
                relative_deviations=(0.75,),
                rejected_count=1,
            )},
            config=VesicleQCConfig(
                curvature=CurvatureQCConfig(threshold=5.0, enabled=True),
                area=AreaQCConfig(max_relative_deviation=0.25, enabled=True),
                localized_deviation=LocalizedDeviationQCConfig(enabled=False),
            ),
        )

    csv_path = tmp_path / "sample.area_qc.csv"
    plot_path = tmp_path / "sample.area_qc.png"
    _write_area_qc_csv(csv_path, FakeEdges())
    _save_area_qc_plot(plot_path, FakeEdges())

    csv_text = csv_path.read_text()
    assert "4,25.0,0.75,True" in csv_text
    assert plot_path.is_file()


def test_minimum_radius_qc_diagnostics_preserve_frame_measurements(tmp_path):
    """Test radius QC writes exact per-frame measurements and flags."""

    class Detection:
        frame_index = 4
        qc = argparse.Namespace(
            diagnostics={"median_radius_pixels": 2.5},
            flags={QCFlag.MINIMUM_RADIUS},
        )

    class FakeEdges:
        successful_detections = (Detection(),)

    csv_path = tmp_path / "sample.minimum_radius_qc.csv"
    _write_minimum_radius_qc_csv(csv_path, FakeEdges())
    csv_text = csv_path.read_text()
    assert "4,2.5,True" in csv_text


def test_readme_cli_and_python_qc_accept_the_same_frames(tmp_path, monkeypatch):
    """Run the documented QC examples on contours that separate both cutoffs."""
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
    commands = re.findall(r"^```bash\n(vesedge qc .*?)\n```", readme, re.M | re.S)
    python_blocks = re.findall(r"^```python\n(.*?)\n```", readme, re.M | re.S)
    python_qc = next(block for block in python_blocks if "permissive_qc =" in block)

    monkeypatch.chdir(tmp_path)
    checkpoint = Path("checkpoints/sample.npz")
    checkpoint.parent.mkdir()
    radii = np.full((3, 120), 10.0)
    radii[1, 0] *= 1.034
    radii[2, 0] *= 1.08
    contours = [ImageContour((0.0, 0.0), row) for row in radii]
    VesicleEdges(
        EdgeExtractionConfig(n_angular_samples=120),
        [EdgeDetection(contour, contour) for contour in contours],
    ).save_checkpoint(checkpoint)

    assert len(commands) == 2
    for command in commands:
        monkeypatch.setattr(sys, "argv", shlex.split(command.replace("\\\n", "")))
        assert vesedge_cli.main() == 0

    namespace = {}
    exec(compile(python_qc, "README.md", "exec"), namespace)
    for directory, python_output, config_name, count in (
        ("qc_standard", "sample.npy", "qc_config", 1),
        ("qc_permissive", "sample_permissive.npy", "permissive_qc", 2),
    ):
        output_dir = Path("results") / directory
        manifest = json.loads((output_dir / "vesedge_qc.json").read_text(encoding="utf-8"))
        assert manifest["qc_config"] == namespace[config_name].to_dict()
        np.testing.assert_array_equal(np.load(output_dir / "sample.npy"), np.load(python_output))
        np.testing.assert_array_equal(np.load(python_output), radii[:count])
