"""Regression tests for explicit VesEdge extraction defaults."""

import sys

import numpy as np
import pytest

from vesmod.VesEdge import EdgeExtractionConfig
from vesmod.VesEdge.checkpoint_io import _extraction_config_from_checkpoint
from vesmod.cli import vesedge_cli


def test_extract_requires_explicit_calibration(monkeypatch):
    """Extraction must not silently assume one pixel per micron."""
    monkeypatch.setattr(sys, "argv", ["vesedge", "extract", "sample.nd2"])

    with pytest.raises(SystemExit):
        vesedge_cli.parse_args()


def test_extract_measured_calibration_uses_api_sampling_default(monkeypatch):
    """CLI and API both default to 120 analysis samples."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "extract",
            "sample.nd2",
            "--pixels-per-micron",
            "13.44",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.pixels_per_micron == pytest.approx(13.44)
    assert not args.assume_one_pixel_per_micron
    assert args.n_angular_samples == EdgeExtractionConfig().n_angular_samples == 120


def test_extract_can_explicitly_assume_unit_calibration(monkeypatch):
    """Unit calibration requires an explicit CLI opt-in."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "extract",
            "sample.nd2",
            "--assume-one-pixel-per-micron",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.assume_one_pixel_per_micron
    assert args.pixels_per_micron is None


def test_extract_native_sampling_is_explicit(monkeypatch):
    """The single sampling option can explicitly retain native sampling."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vesedge",
            "extract",
            "sample.nd2",
            "--pixels-per-micron",
            "13.44",
            "--n-angular-samples",
            "native",
        ],
    )

    args = vesedge_cli.parse_args()

    assert args.n_angular_samples is None


def test_checkpoint_config_restores_calibration_source():
    """New checkpoints retain whether calibration was measured or assumed."""
    checkpoint = {
        "pixels_per_micron": np.asarray(13.44),
        "n_angular_samples": np.asarray(120, dtype=np.int64),
        "calibration_source": np.asarray("measured"),
    }

    config = _extraction_config_from_checkpoint(checkpoint)

    assert config.pixels_per_micron == pytest.approx(13.44)
    assert config.n_angular_samples == 120
    assert config.calibration_source == "measured"


def test_legacy_checkpoint_config_has_unspecified_calibration_source():
    """Older checkpoints remain loadable without invented provenance."""
    checkpoint = {
        "pixels_per_micron": np.asarray(1.0),
        "n_angular_samples": np.asarray(-1, dtype=np.int64),
    }

    config = _extraction_config_from_checkpoint(checkpoint)

    assert config.n_angular_samples is None
    assert config.calibration_source == "unspecified"
