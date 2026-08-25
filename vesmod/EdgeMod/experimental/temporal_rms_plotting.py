"""Plot experimental temporal-RMS population diagnostics."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from .temporal_rms import TemporalRMSResult


def save_temporal_rms_histogram(
    results: Sequence[TemporalRMSResult],
    output_path: str | Path,
) -> None:
    """Save a histogram of successfully measured trajectory amplitudes."""
    if not results:
        return

    amplitudes = [result.amplitude_nm for result in results]
    cutoff_nm = results[0].config.cutoff_nm
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.hist(amplitudes, bins="auto", edgecolor="black")
    if cutoff_nm is not None:
        axis.axvline(
            cutoff_nm,
            color="tab:red",
            linestyle="--",
            label=f"Cutoff: {cutoff_nm:g} nm",
        )
        axis.legend()
    axis.set_xlabel("Absolute temporal RMS amplitude (nm)")
    axis.set_ylabel("Number of spectra")
    axis.set_title("Temporal contour motion")
    figure.savefig(Path(output_path), dpi=150)
    plt.close(figure)
