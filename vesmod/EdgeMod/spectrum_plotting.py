#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composable plots for EdgeMod fluctuation spectra."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .spectrum_utils import HSS97


@dataclass(frozen=True)
class SpectrumPlotData:
    """Inputs required to render one measured spectrum and optional fit."""

    modes: np.ndarray
    avg_amps2: np.ndarray
    fit_result: Any | None = None
    lower_bound: int | None = None
    upper_bound: int | None = None
    lmax: int | None = None
    validation_error: str | None = None


@dataclass(frozen=True)
class SpectrumPlotConfig:
    """Rendering options for composable spectrum plots."""

    nonfit_alpha: float = 0.35
    fit_alpha: float = 1.0
    fitting_region: str = "strip"
    marker: str = "o"
    linewidth: float = 1.5

    def __post_init__(self) -> None:
        if not 0 < self.nonfit_alpha <= 1:
            raise ValueError("nonfit_alpha must be greater than 0 and at most 1.")
        if not 0 < self.fit_alpha <= 1:
            raise ValueError("fit_alpha must be greater than 0 and at most 1.")
        if self.fitting_region not in {"none", "shade", "strip"}:
            raise ValueError("fitting_region must be 'none', 'shade', or 'strip'.")
        if self.linewidth <= 0:
            raise ValueError("linewidth must be positive.")


@dataclass(frozen=True)
class SpectrumPlotResult:
    """Artists created by :func:`plot_spectrum`."""

    figure: plt.Figure
    ax: plt.Axes
    nonfit_artist: Any | None
    fit_data_artist: Any | None
    fit_artist: Any | None
    fitting_region_artist: Any | None


def plot_spectrum(
    data: SpectrumPlotData,
    *,
    ax=None,
    color=None,
    label=None,
    config: SpectrumPlotConfig | None = None,
) -> SpectrumPlotResult:
    """Plot one spectrum on an existing or newly created axis.

    Spectrum identity is represented by ``color``. Fitting status is represented
    by marker opacity and fill, so multiple spectra remain distinguishable when
    overlaid. The fit interval is shown as a compact strip by default, avoiding
    full-height colored bands for overlaid spectra.
    """
    if config is None:
        config = SpectrumPlotConfig()
    modes, measured = _positive_spectrum(data.modes, data.avg_amps2)
    if ax is None:
        _, ax = plt.subplots()
    figure = ax.figure
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]

    has_fit_region = data.lower_bound is not None and data.upper_bound is not None
    selected = _fit_mask(data, modes)
    if not has_fit_region:
        measured_artist = ax.loglog(
            modes,
            measured,
            linestyle="none",
            marker=config.marker,
            color=color,
            alpha=config.fit_alpha,
            label=label,
        )[0]
        fitting_region_artist = None
        return SpectrumPlotResult(
            figure,
            ax,
            measured_artist,
            None,
            None,
            fitting_region_artist,
        )
    nonfit_artist = ax.loglog(
        modes[~selected],
        measured[~selected],
        linestyle="none",
        marker=config.marker,
        markerfacecolor="none",
        markeredgecolor=color,
        color=color,
        alpha=config.nonfit_alpha,
    )[0] if np.any(~selected) else None
    fit_data_artist = ax.loglog(
        modes[selected],
        measured[selected],
        linestyle="none",
        marker=config.marker,
        color=color,
        alpha=config.fit_alpha,
        label=label,
    )[0] if np.any(selected) else None
    fit_artist = None
    if data.fit_result is not None:
        if data.lmax is None:
            raise ValueError("lmax is required when fit_result is provided.")
        if not np.any(selected):
            raise ValueError("fit bounds must select at least one positive mode.")
        predicted = np.asarray(
            HSS97(
                modes[selected],
                data.fit_result.best_values["kC"],
                data.fit_result.best_values["sigma"],
                data.lmax,
            )
        )
        fit_artist = ax.loglog(
            modes[selected],
            predicted,
            color=color,
            linewidth=config.linewidth,
        )[0]

    fitting_region_artist = _plot_fitting_region(
        ax,
        data,
        color,
        config.fitting_region,
    )
    ax.set_xlabel("Fourier mode q")
    ax.set_ylabel(r"$\langle |u_q|^2 \rangle$")
    return SpectrumPlotResult(
        figure,
        ax,
        nonfit_artist,
        fit_data_artist,
        fit_artist,
        fitting_region_artist,
    )


def plot_q4_scaled_spectrum(
    data: SpectrumPlotData,
    *,
    ax=None,
    color=None,
    label=None,
    config: SpectrumPlotConfig | None = None,
):
    """Plot the diagnostic q⁴-scaled spectrum on a caller-owned axis."""
    if config is None:
        config = SpectrumPlotConfig()
    modes, measured = _positive_spectrum(data.modes, data.avg_amps2)
    if ax is None:
        _, ax = plt.subplots()
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]
    has_fit_region = data.lower_bound is not None and data.upper_bound is not None
    selected = _fit_mask(data, modes)
    if not has_fit_region:
        ax.semilogy(
            modes,
            modes**4 * measured,
            linestyle="none",
            marker=config.marker,
            color=color,
            alpha=config.fit_alpha,
            label=label,
        )
        ax.set_xlabel("Fourier mode q")
        ax.set_ylabel(r"$q^4\langle |u_q|^2 \rangle$")
        ax.set_title("q⁴-scaled spectrum")
        return ax.figure, ax
    ax.semilogy(
        modes[~selected],
        modes[~selected] ** 4 * measured[~selected],
        linestyle="none",
        marker=config.marker,
        markerfacecolor="none",
        markeredgecolor=color,
        color=color,
        alpha=config.nonfit_alpha,
    )
    ax.semilogy(
        modes[selected],
        modes[selected] ** 4 * measured[selected],
        linestyle="none",
        marker=config.marker,
        color=color,
        alpha=config.fit_alpha,
        label=label,
    )
    ax.set_xlabel("Fourier mode q")
    ax.set_ylabel(r"$q^4\langle |u_q|^2 \rangle$")
    ax.set_title("q⁴-scaled spectrum")
    return ax.figure, ax


def save_spectrum_fit_diagnostic(data: SpectrumPlotData, path) -> None:
    """Save the spectrum, q⁴-scaled spectrum, and fit residuals."""
    if data.fit_result is None:
        raise ValueError("A fit result is required for a fit diagnostic.")
    modes, measured = _positive_spectrum(data.modes, data.avg_amps2)
    selected = _fit_mask(data, modes)
    predicted = np.asarray(
        HSS97(
            modes[selected],
            data.fit_result.best_values["kC"],
            data.fit_result.best_values["sigma"],
            data.lmax,
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    plot_spectrum(
        data,
        ax=axes[0],
        color="tab:blue",
        config=SpectrumPlotConfig(fitting_region="shade"),
    )
    plot_q4_scaled_spectrum(
        data,
        ax=axes[1],
        color="tab:blue",
        config=SpectrumPlotConfig(fitting_region="none"),
    )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].plot(
        modes[selected],
        (measured[selected] - predicted) / measured[selected],
        "o-",
        color="tab:blue",
    )
    axes[2].set_xlabel("Fitted Fourier mode q")
    axes[2].set_ylabel("(measured - fit) / measured")
    axes[2].set_title("Fit-region residuals")
    figure.suptitle(_diagnostic_title(data.fit_result, data.validation_error))
    output_path = Path(path).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _positive_spectrum(modes, avg_amps2):
    mask = np.asarray(modes) >= 2
    return np.asarray(modes)[mask], np.asarray(avg_amps2)[mask]


def _fit_mask(data, modes):
    if data.lower_bound is None or data.upper_bound is None:
        return np.zeros(modes.shape, dtype=bool)
    return (modes >= data.lower_bound) & (modes < data.upper_bound)


def _plot_fitting_region(ax, data, color, style):
    if style == "none" or data.lower_bound is None or data.upper_bound is None:
        return None
    lower = data.lower_bound - 0.5
    upper = data.upper_bound - 0.5
    if style == "shade":
        return ax.axvspan(lower, upper, color=color, alpha=0.08, zorder=0)
    return ax.axvspan(lower, upper, ymin=0.96, ymax=1.0, color=color, alpha=0.9)


def _diagnostic_title(fit_result, validation_error):
    kc = fit_result.params["kC"]
    sigma = fit_result.params["sigma"]
    kc_stderr = "unknown" if kc.stderr is None else f"{kc.stderr:.3g}"
    sigma_stderr = "unknown" if sigma.stderr is None else f"{sigma.stderr:.3g}"
    parameters = (
        f"kC={kc.value:.4g} ± {kc_stderr}; "
        f"reduced sigma={sigma.value:.4g} ± {sigma_stderr}"
    )
    if validation_error is None:
        return f"Spectrum fit diagnostic\n{parameters}"
    return f"Spectrum fit diagnostic — rejected fit\n{parameters}\n{validation_error}"
