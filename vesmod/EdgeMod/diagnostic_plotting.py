#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic figures for EdgeMod spectrum fitting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .spectrum_utils import HSS97


@dataclass(frozen=True)
class SpectrumDiagnosticData:
    """Inputs required to render one spectrum-fit diagnostic."""

    modes: np.ndarray
    avg_amps2: np.ndarray
    fit_result: Any
    lower_bound: int
    upper_bound: int
    lmax: int
    validation_error: str | None = None


def save_spectrum_fit_diagnostic(
    diagnostic: SpectrumDiagnosticData,
    path,
):
    """Save measured spectrum, compensated spectrum, and fit residuals."""
    plot_modes, measured = _positive_spectrum(
        diagnostic.modes,
        diagnostic.avg_amps2,
    )
    predicted = np.asarray(
        HSS97(
            plot_modes,
            diagnostic.fit_result.best_values["kC"],
            diagnostic.fit_result.best_values["sigma"],
            diagnostic.lmax,
        )
    )
    selected = (
        (plot_modes >= diagnostic.lower_bound)
        & (plot_modes < diagnostic.upper_bound)
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        constrained_layout=True,
    )
    _plot_spectrum(axes[0], plot_modes, measured, predicted, selected)
    _plot_compensated_spectrum(
        axes[1],
        plot_modes,
        measured,
        predicted,
        selected,
    )
    _plot_relative_residuals(
        axes[2],
        plot_modes[selected],
        measured[selected],
        predicted[selected],
    )
    figure.suptitle(
        _diagnostic_title(
            diagnostic.fit_result,
            diagnostic.validation_error,
        )
    )
    output_path = Path(path).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _positive_spectrum(modes, avg_amps2):
    """Return positive modes for which the HSS97 expression is defined."""
    mask = np.asarray(modes) >= 2
    return np.asarray(modes)[mask], np.asarray(avg_amps2)[mask]


def _plot_spectrum(axis, modes, measured, predicted, selected):
    """Plot the measured and attempted theoretical spectra."""
    axis.loglog(modes, measured, "o", color="0.55", label="Measured")
    axis.loglog(
        modes[selected],
        measured[selected],
        "o",
        color="tab:blue",
        label="Fitted modes",
    )
    axis.loglog(modes, predicted, color="tab:orange", label="Attempted fit")
    axis.set_xlabel("Fourier mode q")
    axis.set_ylabel(r"$\langle |u_q|^2 \rangle$")
    axis.set_title("Fluctuation spectrum")
    axis.legend()


def _plot_compensated_spectrum(axis, modes, measured, predicted, selected):
    """Plot q^4-compensated data to expose flattening or noise floors."""
    compensation = modes**4
    axis.semilogy(modes, compensation * measured, "o", color="0.55")
    axis.semilogy(
        modes[selected],
        compensation[selected] * measured[selected],
        "o",
        color="tab:blue",
    )
    axis.semilogy(modes, compensation * predicted, color="tab:orange")
    axis.set_xlabel("Fourier mode q")
    axis.set_ylabel(r"$q^4\langle |u_q|^2 \rangle$")
    axis.set_title(r"$q^4$-compensated spectrum")


def _plot_relative_residuals(axis, modes, measured, predicted):
    """Plot residuals relative to measured fitted-mode amplitudes."""
    residuals = (measured - predicted) / measured
    axis.axhline(0.0, color="black", linewidth=1)
    axis.plot(modes, residuals, "o-", color="tab:red")
    axis.set_xlabel("Fitted Fourier mode q")
    axis.set_ylabel("(measured - fit) / measured")
    axis.set_title("Fit residuals")


def _diagnostic_title(fit_result, validation_error):
    """Return a concise fit-result and validation summary."""
    kc = fit_result.params["kC"]
    sigma = fit_result.params["sigma"]
    kc_stderr = "unknown" if kc.stderr is None else f"{kc.stderr:.3g}"
    sigma_stderr = (
        "unknown" if sigma.stderr is None else f"{sigma.stderr:.3g}"
    )
    parameters = (
        f"kC={kc.value:.4g} ± {kc_stderr}; "
        f"reduced sigma={sigma.value:.4g} ± {sigma_stderr}"
    )
    if validation_error is None:
        return f"Spectrum fit diagnostic\n{parameters}"
    return f"Spectrum fit diagnostic — rejected fit\n{parameters}\n{validation_error}"
