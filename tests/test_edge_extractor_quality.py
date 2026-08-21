#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acceptance tests for VesEdge extraction quality.

These tests intentionally run only the legacy curvature QC check so that
reference acceptance rates remain comparable with values generated before
trajectory-population QC was added.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from vesmod.VesEdge import (
    EdgeExtractionConfig,
    EdgeQCConfig,
    VesicleVideo,
    extract_edge_from_frame,
)
from vesmod.VesEdge.models import (
    EdgeDetection,
    EdgeDetectionFailure,
    QCFlag,
)


def _get_sample_video_paths():
    """Return sorted paths to all sample vesicle-video arrays."""
    test_file_dir = Path(__file__).parent / "sample_vesicle_videos"
    if not test_file_dir.exists():
        pytest.exit(
            f"Test data directory does not exist: {test_file_dir}",
            returncode=1,
        )

    paths = sorted(
        path
        for path in test_file_dir.iterdir()
        if path.suffix == ".npy"
    )
    if not paths:
        pytest.exit(
            f"No files found in test directory: {test_file_dir}",
            returncode=1,
        )
    return paths


def pytest_generate_tests(metafunc):
    """Parameterize tests over all sample vesicle videos."""
    if "filename" not in metafunc.fixturenames:
        return
    paths = _get_sample_video_paths()
    filenames = [path.stem for path in paths]
    metafunc.parametrize("filename", filenames, ids=filenames)


@pytest.fixture(scope="session")
def processed_sample_edges():
    """Return a session cache of successfully processed sample edges."""
    return {}


@pytest.fixture
def sample_edges(filename, processed_sample_edges):
    """Extract and QC the current sample, caching successful runs."""
    if filename in processed_sample_edges:
        return processed_sample_edges[filename]

    path = next(
        path
        for path in _get_sample_video_paths()
        if path.stem == filename
    )
    extraction_config = EdgeExtractionConfig(
        pixels_per_micron=1.0,
        n_angular_samples=None,
    )
    qc_config = EdgeQCConfig(
        curvature_threshold=10.0,
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
        enable_curvature_qc=True,
        enable_population_qc=False,
    )

    video = VesicleVideo(np.load(path))
    edges = video.extract_edges(
        extract_edge_from_frame,
        extraction_config,
    )
    edges.run_qc(qc_config)

    processed_sample_edges[filename] = edges
    return edges


def test_whether_edges_extracted(filename, sample_edges):
    """Verify that edge extraction succeeds for every frame in each sample."""
    failures = [
        result
        for result in sample_edges.detections
        if isinstance(result, EdgeDetectionFailure)
    ]
    assert not failures, (
        f"Edge extraction failed on {len(failures)} frame(s) in {filename}: "
        f"{[failure.error for failure in failures]}"
    )

    for result in sample_edges.detections:
        assert isinstance(result, EdgeDetection)
        assert np.all(np.isfinite(result.full_contour.x))
        assert np.all(np.isfinite(result.full_contour.y))


def test_only_curvature_qc_was_run(filename, sample_edges):
    """Verify that the acceptance test applies only curvature-based QC."""
    for result in sample_edges.detections:
        if isinstance(result, EdgeDetectionFailure):
            continue
        assert result.qc.curvature_score is not None
        assert result.qc.population_label is None
        assert result.qc.population_probability is None
        assert result.qc.flags <= {QCFlag.CURVATURE}


def test_extraction_quality(request, filename, sample_edges):
    """Compare the fraction of usable frames with the stored reference value."""
    n_usable_frames = sum(
        isinstance(result, EdgeDetection) and result.accepted
        for result in sample_edges.detections
    )
    meas_pct_usbl_frames = n_usable_frames / len(sample_edges.detections)

    expected_value_file = (
        Path(__file__).parent
        / "reference_values"
        / f"expected_value_{filename}.json"
    )
    key = "expected pct useable value"

    if request.config.getoption("--update-ref-values"):
        expected_value_file.parent.mkdir(parents=True, exist_ok=True)
        expected_value_file.unlink(missing_ok=True)
        with expected_value_file.open("w") as file:
            json.dump(
                {key: meas_pct_usbl_frames},
                file,
                indent=2,
            )
    else:
        with expected_value_file.open() as file:
            saved_data = json.load(file)
        assert math.isclose(
            meas_pct_usbl_frames,
            saved_data[key],
            abs_tol=0.01,
        ), (
            "Extraction rate does not match "
            f"reference value for {filename}"
        )
