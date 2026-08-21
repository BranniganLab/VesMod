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
    """
    Return the paths to all sample vesicle videos.

    Returns
    -------
    list[Path]
        Sorted paths to the sample ``.npy`` files.

    Raises
    ------
    pytest.exit
        If the sample-video directory does not exist or contains no
        ``.npy`` files.
    """
    test_file_dir = (
        Path(__file__).parent
        / "sample_vesicle_videos"
    )

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


# --------------------------------------------------------------------
# Hook: Make each test run once for each file by using filename as a
# parametrized variable. DO NOT RENAME THIS FUNCTION OR ITS ARGUMENT!
# --------------------------------------------------------------------
def pytest_generate_tests(metafunc):
    """
    Dynamically parameterize tests over all sample vesicle videos.

    Only tests that request the `filename` fixture are parameterized.
    """
    if "filename" not in metafunc.fixturenames:
        return

    paths = _get_sample_video_paths()
    filenames = [path.stem for path in paths]

    metafunc.parametrize(
        "filename",
        filenames,
        ids=filenames,
    )


# ------------------------------------------------------------
# Fixtures: Process each sample lazily and cache successful runs
# ------------------------------------------------------------
@pytest.fixture(scope="session")
def processed_sample_videos():
    """Return a session cache of successfully processed sample videos."""
    return {}


@pytest.fixture
def sample_video(
    filename,
    processed_sample_videos,
):
    """
    Load and process the sample video for the current parameterized case.

    Successfully extracted videos are cached for the rest of the test session
    so each sample is processed at most once. Extraction failures remain
    isolated to the filename that triggered them.

    Returns
    -------
    VesicleVideo
        Processed video corresponding to ``filename``.
    """
    if filename in processed_sample_videos:
        return processed_sample_videos[filename]

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

    video = VesicleVideo(
        np.load(path),
        extraction_config,
        qc_config,
    )
    video.extract_edges(
        extract_edge_from_frame
    )

    processed_sample_videos[filename] = video
    return video


# ----------------------------
# Actual tests
# ----------------------------
def test_whether_edges_extracted(
    filename,
    sample_video,
):
    """
    Verify that edge extraction succeeds for every frame in each sample video.
    """
    video = sample_video

    failures = [
        result
        for result in video.detections
        if isinstance(
            result,
            EdgeDetectionFailure,
        )
    ]

    assert not failures, (
        f"Edge extraction failed on {len(failures)} "
        f"frame(s) in {filename}: "
        f"{[failure.error for failure in failures]}"
    )

    for result in video.detections:
        assert isinstance(
            result,
            EdgeDetection,
        )
        assert np.all(
            np.isfinite(
                result.full_contour.x
            )
        )
        assert np.all(
            np.isfinite(
                result.full_contour.y
            )
        )


def test_only_curvature_qc_was_run(
    filename,
    sample_video,
):
    """
    Verify that the acceptance test applies only curvature-based QC.

    Every successfully detected edge should have a curvature score, while
    population-QC measurements should remain unset.
    """
    video = sample_video

    for result in video.detections:
        if isinstance(
            result,
            EdgeDetectionFailure,
        ):
            continue

        assert (
            result.qc.curvature_score
            is not None
        )
        assert (
            result.qc.population_label
            is None
        )
        assert (
            result.qc.population_probability
            is None
        )
        assert result.qc.flags <= {
            QCFlag.CURVATURE
        }


def test_extraction_quality(
    request,
    filename,
    sample_video,
):
    """
    Compare the fraction of usable frames with the stored reference value.

    A frame is usable when edge extraction succeeded and the resulting
    EdgeDetection passed the enabled QC checks.
    """
    video = sample_video

    n_usable_frames = sum(
        isinstance(result, EdgeDetection)
        and result.accepted
        for result in video.detections
    )

    meas_pct_usbl_frames = (
        n_usable_frames
        / len(video.detections)
    )

    expected_value_file = (
        Path(__file__).parent
        / "reference_values"
        / f"expected_value_{filename}.json"
    )

    key = "expected pct useable value"

    if request.config.getoption(
        "--update-ref-values"
    ):
        expected_value_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        expected_value_file.unlink(
            missing_ok=True
        )

        new_reference_value_dict = {
            key: meas_pct_usbl_frames
        }

        with expected_value_file.open(
            "w"
        ) as file:
            json.dump(
                new_reference_value_dict,
                file,
                indent=2,
            )

    else:
        with expected_value_file.open() as file:
            saved_data = json.load(file)

        exp_pct_usbl_frames = (
            saved_data[key]
        )

        assert math.isclose(
            meas_pct_usbl_frames,
            exp_pct_usbl_frames,
            abs_tol=0.01,
        ), (
            "Extraction rate does not match "
            f"reference value for {filename}"
        )
