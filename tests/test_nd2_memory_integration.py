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


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux /proc RSS accounting")
def test_real_nd2_sequential_reads_have_bounded_file_backed_rss():
    """Touching many real ND2 frames must not produce linear RssFile growth.

    Set ``VESMOD_ND2_MEMORY_TEST`` to a representative ND2 acquisition. The
    workload runs in a fresh Python process, touches every pixel, and spans five
    reader-memory-budget windows. Comparing the first and last windows avoids
    making the assertion depend on the parent pytest process or its allocator
    state.
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

    worker = textwrap.dedent(
        """
        import json
        import math
        import sys

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
        with ND2FrameSource(path) as source:
            first = source[0]
            np.sum(first)
            frame_bytes = first.nbytes
            frames_per_window = max(
                1,
                math.ceil(_ND2_READER_MEMORY_BUDGET_BYTES / frame_bytes),
            )
            required_frames = 5 * frames_per_window
            if len(source) < required_frames:
                print(
                    json.dumps(
                        {
                            "skip": True,
                            "available_frames": len(source),
                            "required_frames": required_frames,
                            "frames_per_window": frames_per_window,
                        }
                    )
                )
                raise SystemExit(0)

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
                    "frame_bytes": frame_bytes,
                    "frames_per_window": frames_per_window,
                    "window_peaks": window_peaks,
                    "budget_bytes": _ND2_READER_MEMORY_BUDGET_BYTES,
                }
            )
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", worker, str(nd2_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    if result["skip"]:
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
