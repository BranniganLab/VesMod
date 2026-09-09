"""Tests for recorded VesEdge QC replay."""

import json

import numpy as np
import pytest

from vesmod.VesEdge import (
    EdgeQCConfig,
    RecordedQCSelection,
    load_recorded_qc,
    replay_recorded_qc,
)
from vesmod.VesEdge.experimental.internal_vesicle_qc import (
    InternalVesicleQCConfig,
)


def _selection(tmp_path, checkpoint, config=None):
    config = config or EdgeQCConfig(curvature_threshold=5.0)
    provenance_path = tmp_path / "qc" / "vesedge_qc.json"
    provenance_path.parent.mkdir()
    provenance_path.write_text(json.dumps({
        "checkpoint_manifest": [str(checkpoint.resolve())],
        "qc_config": config.to_dict(),
    }))
    return load_recorded_qc(provenance_path, [checkpoint])


def test_load_recorded_qc_rejects_manifest_mismatch(tmp_path):
    """Requested checkpoints must belong to the recorded QC batch."""
    recorded = tmp_path / "recorded.npz"
    requested = tmp_path / "requested.npz"
    selection = _selection(tmp_path, recorded)

    with pytest.raises(ValueError, match=str(requested.resolve())):
        selection.validate_checkpoints([requested])


def test_replay_does_not_supply_frames_to_image_independent_qc(tmp_path):
    """Image-independent replay does not require or supply a frame source."""
    checkpoint = tmp_path / "sample.npz"
    selection = _selection(tmp_path, checkpoint)

    class Edges:
        qc_result = None
        accepted_detections = [object()]

        def run_qc(self, config):
            assert config is selection.config
            self.qc_result = object()

    result = replay_recorded_qc(Edges(), checkpoint, selection)

    assert result.accepted_count == 1
    assert not result.all_rejected


def test_replay_supplies_frames_to_image_dependent_qc(tmp_path):
    """Image-dependent replay receives the caller's already-opened frames."""
    checkpoint = tmp_path / "sample.npz"
    config = EdgeQCConfig(
        internal_vesicle=InternalVesicleQCConfig(enabled=True)
    )
    selection = _selection(tmp_path, checkpoint, config)
    frames = np.zeros((1, 4, 4))

    class Edges:
        qc_result = None
        accepted_detections = [object()]

        def run_qc(self, supplied_config, supplied_frames):
            assert supplied_config is selection.config
            assert supplied_frames is frames
            self.qc_result = object()

    replay_recorded_qc(Edges(), checkpoint, selection, frames=frames)


def test_replay_preserves_failure_and_all_rejected_distinction(tmp_path):
    """Only a completed replay with no accepted frames is a valid rejection."""
    checkpoint = tmp_path / "sample.npz"
    selection = _selection(tmp_path, checkpoint)

    class FailedEdges:
        qc_result = None
        accepted_detections = []

        def run_qc(self, config):
            raise ValueError("replay failed")

    with pytest.raises(ValueError, match="replay failed"):
        replay_recorded_qc(FailedEdges(), checkpoint, selection)

    class RejectedEdges(FailedEdges):
        def run_qc(self, config):
            self.qc_result = object()
            raise ValueError("no frames passed quality control")

    result = replay_recorded_qc(RejectedEdges(), checkpoint, selection)
    assert result.all_rejected
    assert result.accepted_count == 0


def test_replay_does_not_accept_restored_stale_all_rejected_result(tmp_path):
    """A failed replay cannot reuse a prior all-rejected result as success."""
    checkpoint = tmp_path / "sample.npz"
    selection = _selection(tmp_path, checkpoint)
    previous_result = object()

    class Edges:
        qc_result = previous_result
        accepted_detections = []

        def run_qc(self, config):
            raise ValueError("check failed before producing a result")

    with pytest.raises(ValueError, match="check failed"):
        replay_recorded_qc(Edges(), checkpoint, selection)


def test_paired_verification_requires_output_for_accepted_frames(tmp_path):
    """Strict replay reports a checkpoint-specific missing paired output."""
    checkpoint = tmp_path / "inputs" / "sample.npz"
    selection = _selection(tmp_path, checkpoint)

    class Edges:
        qc_result = object()
        accepted_detections = [object()]

        def run_qc(self, config):
            pass

    with pytest.raises(FileNotFoundError, match=str(checkpoint.resolve())):
        replay_recorded_qc(
            Edges(),
            checkpoint,
            selection,
            input_root=checkpoint.parent,
            verify_paired_output=True,
        )


@pytest.mark.parametrize("saved", [np.ones((1, 2)), np.zeros((2, 2))])
def test_paired_verification_rejects_shape_or_value_mismatch(tmp_path, saved):
    """Strict replay detects both shape and numeric paired-array mismatches."""
    checkpoint = tmp_path / "inputs" / "sample.npz"
    selection = _selection(tmp_path, checkpoint)
    np.save(selection.output_dir / "sample.npy", saved)

    class Edges:
        qc_result = object()
        accepted_detections = [object(), object()]
        accepted_radii_microns = np.ones((2, 2))

        def run_qc(self, config):
            pass

    with pytest.raises(ValueError, match=str(checkpoint.resolve())):
        replay_recorded_qc(
            Edges(),
            checkpoint,
            selection,
            input_root=checkpoint.parent,
            verify_paired_output=True,
        )
