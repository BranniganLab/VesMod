"""Integration coverage for bounded resident memory during real ND2 reads.

This test intentionally requires a real ND2 acquisition because fake readers and
NumPy fixtures cannot reproduce the mmap/demand-paging behavior that caused
file-backed RSS to grow with every sequential frame.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


_ND2_MEMORY_TEST_ENV = "VESMOD_ND2_MEMORY_TEST"
_ND2_MEMORY_TEST_AXIS_SELECTION_ENV = "VESMOD_ND2_MEMORY_TEST_AXIS_SELECTION"


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux /proc RSS accounting")
def test_real_nd2_sequential_reads_have_bounded_file_backed_rss():
    """Touching many real ND2 frames must not produce linear RssFile growth.

    Set ``VESMOD_ND2_MEMORY_TEST`` to a representative ND2 acquisition. The
    optional ``VESMOD_ND2_MEMORY_TEST_AXIS_SELECTION`` is a JSON object mapping
    non-time axes such as ``P``, ``Z``, or ``C`` to selected indices. The
    workload runs in a fresh Python process, touches every pixel, and spans five
    reader-memory-budget windows.
    """
    configured_path = os.environ.get(_ND2_MEMORY_TEST_ENV)
    if configured_path is None:
        pytest.skip(
            f"set {_ND2_MEMORY_TEST_ENV} to a real ND2 acquisition to run the "
            "ND2 resident-memory regression test"
        )

    nd2_path = Path(configured_path).expanduser().resolve()
    if not nd2_path.is_file():
        pytest.fail(f"{_ND2_MEMORY_TEST_ENV} does not exist: {nd2_path}")

    configured_selection = os.environ.get(
        _ND2_MEMORY_TEST_AXIS_SELECTION_ENV,
        "{}",
    )
    try:
        parsed_selection = json.loads(configured_selection)
        if not isinstance(parsed_selection, dict):
            raise TypeError("axis selection must be a JSON object")
        axis_selection = {
            str(axis).upper(): int(index)
            for axis, index in parsed_selection.items()
        }
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        pytest.fail(
            f"{_ND2_MEMORY_TEST_AXIS_SELECTION_ENV} must be a JSON object "
            f"mapping axis names to integer indices: {exc}"
        )

    worker = textwrap.dedent(
        """
        import json
        import math
        import sys

        import nd2
        import numpy as np

        from vesmod.VesEdge.frame_source import (
            ND2FrameSource,
            _ND2_READER_MEMORY_BUDGET_BYTES,
        )


        def rss_file_bytes():
            with open("/proc/self/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("RssFile:"):
                        return int(line.split()[1]) * 1024
            raise RuntimeError("/proc/self/status does not report RssFile")


        path = sys.argv[1]
        axis_selection = json.loads(sys.argv[2])

        with nd2.ND2File(path) as nd2_file:
            ambiguous_axes = [
                axis
                for axis, size in nd2_file.sizes.items()
                if axis not in {"T", "Y", "X"}
                and size > 1
                and axis not in axis_selection
            ]
        if ambiguous_axes:
            print(
                json.dumps(
                    {
                        "skip": True,
                        "reason": "ambiguous_axes",
                        "axes": ambiguous_axes,
                    }
                )
            )
            raise SystemExit(0)

        with ND2FrameSource(path, axis_selection=axis_selection) as probe:
            first = probe[0]
            np.sum(first)
            bytes_per_read = probe._bytes_since_reopen
            available_frames = len(probe)

        frames_per_window = max(
            1,
            math.ceil(_ND2_READER_MEMORY_BUDGET_BYTES / bytes_per_read),
        )
        required_frames = 5 * frames_per_window
        if available_frames < required_frames:
            print(
                json.dumps(
                    {
                        "skip": True,
                        "reason": "too_short",
                        "available_frames": available_frames,
                        "required_frames": required_frames,
                        "frames_per_window": frames_per_window,
                    }
                )
            )
            raise SystemExit(0)

        with ND2FrameSource(path, axis_selection=axis_selection) as source:
            window_peaks = []
            for window in range(5):
                peak = 0
                start = window * frames_per_window
                stop = start + frames_per_window
                for index in range(start, stop):
                    frame = source[index]
                    np.sum(frame)
                    peak = max(peak, rss_file_bytes())
                window_peaks.append(peak)

        print(
            json.dumps(
                {
                    "skip": False,
                    "bytes_per_read": bytes_per_read,
                    "frames_per_window": frames_per_window,
                    "window_peaks": window_peaks,
                    "budget_bytes": _ND2_READER_MEMORY_BUDGET_BYTES,
                }
            )
        )
        """
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            worker,
            str(nd2_path),
            json.dumps(axis_selection),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    if result["skip"]:
        if result["reason"] == "ambiguous_axes":
            axes = ", ".join(result["axes"])
            pytest.skip(
                "ND2 acquisition has unsupported ambiguous non-time axes "
                f"({axes}); set {_ND2_MEMORY_TEST_AXIS_SELECTION_ENV} to a "
                "JSON selection such as '{\"Z\": 0, \"C\": 0}'"
            )
        pytest.skip(
            "ND2 acquisition is too short to exercise five reader-memory "
            f"windows: {result['available_frames']} frames available, "
            f"{result['required_frames']} required"
        )

    peaks = result["window_peaks"]
    budget = result["budget_bytes"]
    late_growth = peaks[-1] - peaks[0]

    # The historical failure accumulated approximately one additional reader
    # window of file-backed RSS during each sequential window. Five windows
    # therefore produced several budgets of growth. Allow two full budgets of
    # drift here so normal filesystem/cache noise does not make the test brittle
    # while still rejecting that linear-growth behavior.
    assert late_growth <= 2 * budget, (
        "file-backed RSS continued growing across ND2 reader recycle windows: "
        f"window peaks={peaks}, late growth={late_growth / 1024**2:.1f} MiB, "
        f"allowed={2 * budget / 1024**2:.1f} MiB"
    )
