"""End-to-end scientific acceptance test for the VesMod pipeline.

This test deliberately crosses VesMod's two durable hand-off formats:
VesEdge's extraction checkpoint (``.npz``) and its QC-filtered contour array
(``.npy``). It therefore detects drift in extraction, checkpoint round trips,
QC, spectrum construction, and physical fitting.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vesmod.EdgeMod import Spectrum, SpectrumFitConfig
from vesmod.VesEdge import (
    EdgeDetection,
    EdgeExtractionConfig,
    EdgeQCConfig,
    VesicleEdges,
    VesicleVideo,
    extract_edge_from_frame,
)


TEST_ROOT = Path(__file__).parent
ACCEPTANCE_CASES = (
    ("ND_Acquisition_2_crop", "reference_video_1"),
)
REFERENCE_SCHEMA_VERSION = 1

EXTRACTION_CONFIG = EdgeExtractionConfig(
    pixels_per_micron=13.44,
    n_angular_samples=120,
)
QC_CONFIG = EdgeQCConfig(
    curvature_threshold=5.0,
    enable_curvature_qc=True,
    max_relative_area_deviation=0.25,
    enable_area_qc=True,
)
FIT_CONFIG = SpectrumFitConfig(
    lower_bound=3,
    upper_bound=8,
    lmax=500,
    free_sigma=True,
    temperature=295.0,
)


def _fixture_path(case_name: str) -> Path:
    """Return the input video-array fixture for one acceptance case."""
    return TEST_ROOT / "sample_vesicle_videos" / f"{case_name}.npy"


def _reference_path(reference_name: str) -> Path:
    """Return the canonical scientific outputs for one acceptance case."""
    return TEST_ROOT / "acceptance" / f"{reference_name}.npz"


def _sha256(path: Path) -> str:
    """Return a stable identity for the input fixture bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _successful_arrays(edges: VesicleEdges) -> dict[str, np.ndarray]:
    """Collect canonical arrays at the extraction boundary."""
    detections = edges.successful_detections
    return {
        "extraction_successful_frame_indices": np.asarray(
            [detection.frame_index for detection in detections],
            dtype=np.int64,
        ),
        "extraction_origins_pixels": np.asarray(
            [detection.full_contour.origin for detection in detections],
            dtype=float,
        ),
        "extraction_analysis_radii_pixels": np.stack(
            [detection.analysis_contour.r for detection in detections]
        ),
    }


def _run_pipeline(case_name: str, tmp_path: Path) -> dict[str, np.ndarray]:
    """Run all stable public pipeline stages and collect their outputs."""
    input_path = _fixture_path(case_name)
    video = VesicleVideo(np.load(input_path, allow_pickle=False))
    video.source_path = input_path
    edges = video.extract_edges(extract_edge_from_frame, EXTRACTION_CONFIG)

    results = _successful_arrays(edges)
    results["extraction_failed_frame_indices"] = np.asarray(
        [
            result.frame_index
            for result in edges.detections
            if not isinstance(result, EdgeDetection)
        ],
        dtype=np.int64,
    )

    checkpoint_path = tmp_path / f"{case_name}.npz"
    edges.save_checkpoint(checkpoint_path)
    edges = VesicleEdges.from_checkpoint(checkpoint_path)
    edges.run_qc(QC_CONFIG)

    successful = edges.successful_detections
    area_result = edges.qc_result.area
    results.update(
        {
            "qc_curvature_scores": np.asarray(
                edges.qc_result.curvature.scores,
                dtype=float,
            ),
            "qc_areas_pixels2": np.asarray(
                area_result.areas_pixels2,
                dtype=float,
            ),
            "qc_relative_area_deviations": np.asarray(
                area_result.relative_deviations,
                dtype=float,
            ),
            "qc_accepted_frame_indices": np.asarray(
                [
                    detection.frame_index
                    for detection in successful
                    if detection.qc.passed
                ],
                dtype=np.int64,
            ),
            "qc_accepted_radii_microns": edges.accepted_radii_microns,
        }
    )

    accepted_path = tmp_path / f"{case_name}.npy"
    edges.save_edge_to_npy(accepted_path)
    spectrum = Spectrum(accepted_path)
    fit = spectrum.extract_kc_from_fit(FIT_CONFIG)
    results.update(
        {
            "spectrum_r0_microns": np.asarray(spectrum.r0),
            "spectrum_modes": spectrum.modes,
            "spectrum_avg_amps2": spectrum.avg_amps2,
            "fit_kc_kbt": np.asarray(fit.kC),
            "fit_surface_tension_newtons_per_meter": np.asarray(
                fit.surface_tension
            ),
        }
    )

    metadata = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "input_file": input_path.name,
        "input_sha256": _sha256(input_path),
        "extraction_config": asdict(EXTRACTION_CONFIG),
        "qc_config": asdict(QC_CONFIG),
        "fit_config": asdict(FIT_CONFIG),
    }
    results["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    return results


def _assert_matches_reference(
    measured: dict[str, np.ndarray],
    reference_path: Path,
) -> None:
    """Compare discrete outputs exactly and floating outputs numerically."""
    if not reference_path.is_file():
        pytest.fail(
            f"Missing acceptance reference: {reference_path}. Run this test "
            "with --update-acceptance-reference and review the new artifact."
        )

    with np.load(reference_path, allow_pickle=False) as stored:
        expected = {key: stored[key] for key in stored.files}

    assert measured.keys() == expected.keys()
    exact_keys = {
        "metadata_json",
        "extraction_successful_frame_indices",
        "extraction_failed_frame_indices",
        "qc_accepted_frame_indices",
        "spectrum_modes",
    }
    for key in exact_keys:
        np.testing.assert_array_equal(
            measured[key],
            expected[key],
            err_msg=f"Acceptance checkpoint changed: {key}",
        )

    fit_keys = {
        "fit_kc_kbt",
        "fit_surface_tension_newtons_per_meter",
    }
    for key in measured.keys() - exact_keys - fit_keys:
        np.testing.assert_allclose(
            measured[key],
            expected[key],
            rtol=1e-7,
            atol=1e-10,
            err_msg=f"Acceptance checkpoint changed: {key}",
        )

    for key in fit_keys:
        np.testing.assert_allclose(
            measured[key],
            expected[key],
            rtol=1e-5,
            atol=1e-12,
            err_msg=f"Acceptance checkpoint changed: {key}",
        )


@pytest.mark.parametrize(
    ("case_name", "reference_name"),
    ACCEPTANCE_CASES,
    ids=[reference_name for _, reference_name in ACCEPTANCE_CASES],
)
def test_end_to_end_acceptance(
    request,
    tmp_path,
    case_name,
    reference_name,
):
    """Require the full scientific pipeline to match reviewed output."""
    measured = _run_pipeline(case_name, tmp_path)
    reference_path = _reference_path(reference_name)
    if request.config.getoption("--update-acceptance-reference"):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(reference_path, **measured)
        return

    _assert_matches_reference(measured, reference_path)
