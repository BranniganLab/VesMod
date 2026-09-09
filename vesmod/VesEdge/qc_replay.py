"""Replay VesEdge quality control recorded by the QC CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from vesmod.io import map_output_path

from .frame_source import FrameSource
from .qc_checks import QC_CHECKS
from .qc_config import EdgeQCConfig
from .vesicle_edges import VesicleEdges


@dataclass(frozen=True)
class RecordedQCSelection:
    """Validated configuration and checkpoint selection from QC provenance."""

    provenance_path: Path
    config: EdgeQCConfig
    checkpoint_manifest: frozenset[Path]

    @property
    def output_dir(self) -> Path:
        """Return the directory containing the recorded QC outputs."""
        return self.provenance_path.parent

    @property
    def requires_frames(self) -> bool:
        """Return whether any enabled recorded check consumes source images."""
        return any(
            check.requires_frames
            and check.enabled(self.config.for_check(check.name))
            for check in QC_CHECKS
        )

    def validate_checkpoints(self, checkpoints: Iterable[Path]) -> None:
        """Require every requested checkpoint to be in the recorded manifest."""
        unselected = [
            path.resolve()
            for path in checkpoints
            if path.resolve() not in self.checkpoint_manifest
        ]
        if unselected:
            names = ", ".join(str(path) for path in unselected)
            raise ValueError(
                "Selected checkpoint(s) are not present in the QC manifest: "
                f"{names}"
            )


@dataclass(frozen=True)
class QCReplayResult:
    """Outcome of a successful replay, including valid all-rejected results."""

    accepted_count: int
    all_rejected: bool
    paired_output: Path | None = None


def load_recorded_qc(
    path: Path,
    checkpoints: Iterable[Path] = (),
) -> RecordedQCSelection:
    """Load QC provenance and validate its configuration and manifest."""
    provenance_path = path.expanduser().resolve()
    if provenance_path.is_dir():
        provenance_path = provenance_path / "vesedge_qc.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"QC provenance does not exist: {provenance_path}"
        )

    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        config = EdgeQCConfig.from_dict(provenance["qc_config"])
        manifest = frozenset(
            Path(item).expanduser().resolve()
            for item in provenance["checkpoint_manifest"]
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid VesEdge QC provenance: {provenance_path}"
        ) from error

    selection = RecordedQCSelection(provenance_path, config, manifest)
    selection.validate_checkpoints(checkpoints)
    return selection


def replay_recorded_qc(
    edges: VesicleEdges,
    checkpoint: Path,
    selection: RecordedQCSelection,
    *,
    frames: FrameSource | np.ndarray | None = None,
    input_root: Path | None = None,
    verify_paired_output: bool = False,
) -> QCReplayResult:
    """Replay recorded QC and optionally verify its filtered NumPy output."""
    checkpoint = checkpoint.resolve()
    selection.validate_checkpoints((checkpoint,))
    if selection.requires_frames and frames is None:
        raise ValueError(
            f"Recorded QC for {checkpoint} requires source video frames."
        )

    previous_result = edges.qc_result
    try:
        if selection.requires_frames:
            edges.run_qc(selection.config, frames)
        else:
            edges.run_qc(selection.config)
    except ValueError:
        if (
            edges.qc_result is previous_result
            or edges.qc_result is None
            or edges.accepted_detections
        ):
            raise

    accepted_count = len(edges.accepted_detections)
    all_rejected = accepted_count == 0
    if not verify_paired_output or all_rejected:
        return QCReplayResult(accepted_count, all_rejected)
    if input_root is None:
        raise ValueError("input_root is required to verify paired QC output.")

    paired_output = map_output_path(
        checkpoint,
        input_root,
        selection.output_dir,
        suffix=".npy",
    )
    if not paired_output.is_file():
        raise FileNotFoundError(
            f"No paired QC .npy exists for {checkpoint}: {paired_output}"
        )
    saved_radii = np.load(paired_output, allow_pickle=False)
    reconstructed = edges.accepted_radii_microns
    if saved_radii.shape != reconstructed.shape or not np.allclose(
        saved_radii,
        reconstructed,
        equal_nan=True,
    ):
        raise ValueError(
            f"Paired QC output does not match {checkpoint}: {paired_output}"
        )
    return QCReplayResult(accepted_count, False, paired_output)
