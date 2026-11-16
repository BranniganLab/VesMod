#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composable plots for EdgeMod fluctuation spectra."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SpectrumPlotData:
    """Inputs required to render one measured spectrum and optional fit."""

    modes: np.ndarray
    avg_amps2: np.ndarray
    fit_result: Any | None = None
    lower_bound: int | None = None
    upper_bound: int | None = None
    lmax: int | None = None
    exposure_time: float = 0.0


@dataclass(frozen=True)
class SpectrumPlotConfig:
    """Rendering options for composable spectrum plots."""

    nonfit_alpha: float = 0.35
    fit_alpha: float = 1.0
    fitting_region: str = "strip"
    marker: str = "o"
    linewidth: float = 1.5
    include_q1: bool = False
    power: str = "measured"

    def __post_init__(self) -> None:
        """Validate plot configuration.

        Returns
        -------
        None
            Raises an exception for invalid plot options.
        """
        if self.power not in {"measured", "corrected", "both"}:
            raise ValueError("power must be measured, corrected, or both.")
        if not 0 < self.nonfit_alpha <= 1:
            raise ValueError("nonfit_alpha must be greater than 0 and at most 1.")
        if not 0 < self.fit_alpha <= 1:
            raise ValueError("fit_alpha must be greater than 0 and at most 1.")
        if self.fitting_region not in {"none", "shade", "strip"}:
            raise ValueError("fitting_region must be 'none', 'shade', or 'strip'.")
        if self.linewidth <= 0:
            raise ValueError("linewidth must be positive.")
        if not isinstance(self.include_q1, bool):
            raise TypeError("include_q1 must be a bool.")


@dataclass(frozen=True)
class SpectrumPlotResult:
    """Artists created by :func:`plot_spectrum`."""

    figure: plt.Figure
    ax: plt.Axes
    nonfit_artist: Any | None
    fit_data_artist: Any | None
    fit_artist: Any | None
    fitting_region_artist: Any | None
    corrected_nonfit_artist: Any | None = None
    corrected_fit_data_artist: Any | None = None
    corrected_fit_artist: Any | None = None


def plot_spectrum(data, *, ax=None, color=None, label=None, config=None):
    """Plot measured powers, camera-corrected powers, or both.

    Corrected powers use the fitted instantaneous/camera power ratio. They
    are model-dependent estimates for display, not inputs to another fit.
    Correction requires a camera fit and is shown for 2 <= q <= lmax.

    Parameters
    ----------
    data : SpectrumPlotData
        Measured spectrum and optional fit.
    ax : matplotlib.axes.Axes or None
        Target axes, or None to create axes.
    color : matplotlib color or None
        Series color, or None for the default cycle.
    label : str or None
        Optional series label.
    config : SpectrumPlotConfig or None
        Plot settings, or None for defaults.

    Returns
    -------
    SpectrumPlotResult
        Figure, axes, and measured or corrected artists.
    """
    if config is None:
        config = SpectrumPlotConfig()
    if config.power == "measured":
        return _plot_spectrum(data, ax=ax, color=color, label=label, config=config)
    if data.fit_result is None or data.exposure_time <= 0 or data.lmax is None:
        raise ValueError("Corrected powers require an existing camera integration fit.")
    if config.include_q1:
        raise ValueError("Camera correction is defined only for q >= 2.")
    mask = (data.modes >= 2) & (data.modes <= data.lmax)
    modes = data.modes[mask]
    evaluate_instantaneous = getattr(data.fit_result, "eval_instantaneous", None)
    if evaluate_instantaneous is None:
        raise ValueError("Corrected powers require an instantaneous prediction from the fitted model.")
    instantaneous = np.asarray(evaluate_instantaneous(modes))
    camera = np.asarray(data.fit_result.eval(q=modes))
    if np.any(~np.isfinite(camera)) or np.any(camera <= 0):
        raise ValueError("Camera prediction must be finite and positive for correction.")
    corrected = replace(data, modes=modes, avg_amps2=data.avg_amps2[mask] * instantaneous / camera)
    if ax is None:
        _, ax = plt.subplots()
    if color is None:
        color = ax._get_lines.get_next_color()
    def series_label(kind):
        """Series label.

        Parameters
        ----------
        kind : str
            Label for the power view.

        Returns
        -------
        str
            Label for the plotted data series.
        """
        return f"{label} ({kind})" if label else kind.capitalize()
    measured_result = None
    if config.power == "both":
        measured_result = _plot_spectrum(
            data, ax=ax, color=color, label=series_label("measured"), config=config)
    corrected_result = _plot_spectrum(
        corrected, ax=ax, color=color, label=series_label("camera-corrected"),
        config=replace(config, marker="s",
                       fitting_region="none" if measured_result else config.fitting_region),
        instantaneous=instantaneous)
    result = corrected_result if measured_result is None else measured_result
    return replace(result,
                   corrected_nonfit_artist=corrected_result.nonfit_artist,
                   corrected_fit_data_artist=corrected_result.fit_data_artist,
                   corrected_fit_artist=corrected_result.fit_artist)


def _plot_spectrum(
    data: SpectrumPlotData,
    *,
    ax=None,
    color=None,
    label=None,
    config: SpectrumPlotConfig | None = None,
    instantaneous=None,
) -> SpectrumPlotResult:
    """Plot one spectrum on an existing or newly created axis.

    Spectrum identity is represented by ``color``. Fitting status is represented
    by marker opacity and fill, so multiple spectra remain distinguishable when
    overlaid. The fit interval is shown as a compact strip by default, avoiding
    full-height colored bands for overlaid spectra.

    Parameters
    ----------
    data : SpectrumPlotData
        Measured spectrum and optional fit.
    ax : matplotlib.axes.Axes or None
        Target axes, or None to create axes.
    color : matplotlib color or None
        Series color, or None for the default cycle.
    label : str or None
        Optional series label.
    config : SpectrumPlotConfig or None
        Plot settings, or None for defaults.
    instantaneous : numpy.ndarray or None
        Optional instantaneous powers aligned with the plotted positive modes.

    Returns
    -------
    SpectrumPlotResult
        Figure, axes, and artists for one spectrum.
    """
    if config is None:
        config = SpectrumPlotConfig()
    modes, measured = _positive_spectrum(data.modes, data.avg_amps2, config.include_q1)
    if ax is None:
        _, ax = plt.subplots()
    figure = ax.figure
    if color is None:
        color = ax._get_lines.get_next_color()

    has_fit_region = data.lower_bound is not None and data.upper_bound is not None
    if data.fit_result is not None and not has_fit_region:
        raise ValueError(
            "lower_bound and upper_bound are required when fit_result is provided."
        )
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
            instantaneous[selected]
            if instantaneous is not None else data.fit_result.eval(q=modes[selected])
        )
        fit_artist = ax.loglog(
            modes[selected],
            predicted,
            color=color,
            linewidth=config.linewidth,
            linestyle="--" if instantaneous is not None else "-",
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


def plot_q3_scaled_spectrum(
    data: SpectrumPlotData,
    *,
    ax=None,
    color=None,
    label=None,
    config: SpectrumPlotConfig | None = None,
):
    """Plot the diagnostic q³-scaled spectrum on a caller-owned axis.

    Parameters
    ----------
    data : SpectrumPlotData
        Measured spectrum and optional fit.
    ax : matplotlib.axes.Axes or None
        Target axes, or None to create axes.
    color : matplotlib color or None
        Series color, or None for the default cycle.
    label : str or None
        Optional series label.
    config : SpectrumPlotConfig or None
        Plot settings, or None for defaults.

    Returns
    -------
    tuple
        Figure and axes with q³-scaled powers.
    """
    result = plot_spectrum(data, ax=ax, color=color, label=label, config=config)
    artists = (result.nonfit_artist, result.fit_data_artist, result.fit_artist,
               result.corrected_nonfit_artist, result.corrected_fit_data_artist,
               result.corrected_fit_artist)
    # Corrected-only views expose the same artist in both result fields.
    for artist in {artist for artist in artists if artist is not None}:
        q = np.asarray(artist.get_xdata())
        artist.set_ydata(q**3 * np.asarray(artist.get_ydata()))
    result.ax.set_xscale("linear")
    result.ax.relim()
    result.ax.autoscale_view()
    result.ax.set_ylabel(r"$q^3\langle |u_q|^2 \rangle$")
    result.ax.set_title("q³-scaled spectrum")
    return result.figure, result.ax


def save_spectrum_fit_diagnostic(data: SpectrumPlotData, path) -> None:
    """Save the spectrum, q³-scaled spectrum, and fit residuals.

    Parameters
    ----------
    data : SpectrumPlotData
        Spectrum and fit to visualize.
    path : str or pathlib.Path
        Destination image path.

    Returns
    -------
    None
        Writes the diagnostic figure.
    """
    if data.fit_result is None:
        raise ValueError("A fit result is required for a fit diagnostic.")
    modes, measured = _positive_spectrum(data.modes, data.avg_amps2)
    selected = _fit_mask(data, modes)
    predicted = np.asarray(
        data.fit_result.eval(q=modes[selected])
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    plot_spectrum(
        data,
        ax=axes[0],
        color="tab:blue",
        config=SpectrumPlotConfig(fitting_region="shade"),
    )
    plot_q3_scaled_spectrum(
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
    figure.suptitle(_diagnostic_title(data.fit_result))
    output_path = Path(path).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _positive_spectrum(modes, avg_amps2, include_q1=False):
    """Positive spectrum.

    Parameters
    ----------
    modes : numpy.ndarray
        Fourier modes.
    avg_amps2 : array-like
        Powers aligned with modes.
    include_q1 : bool
        Whether to retain the first positive mode.

    Returns
    -------
    tuple
        Selected modes and aligned powers.
    """
    mask = np.asarray(modes) >= (1 if include_q1 else 2)
    return np.asarray(modes)[mask], np.asarray(avg_amps2)[mask]


def _fit_mask(data, modes):
    """Fit mask.

    Parameters
    ----------
    data : SpectrumPlotData
        Spectrum plot inputs containing the fit bounds.
    modes : numpy.ndarray
        Modes to check against the bounds.

    Returns
    -------
    numpy.ndarray
        Boolean mask identifying the fitted modes.
    """
    if data.lower_bound is None or data.upper_bound is None:
        return np.zeros(modes.shape, dtype=bool)
    return (modes >= data.lower_bound) & (modes < data.upper_bound)


def _plot_fitting_region(ax, data, color, style):
    """Plot fitting region.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.
    data : SpectrumPlotData
        Plot inputs containing the fit bounds.
    color : matplotlib color
        Color of the region marker.
    style : str
        Region display style.

    Returns
    -------
    Artist or None
        Region marker when drawn.
    """
    if style == "none" or data.lower_bound is None or data.upper_bound is None:
        return None
    lower = data.lower_bound - 0.5
    upper = data.upper_bound - 0.5
    if style == "shade":
        return ax.axvspan(lower, upper, color=color, alpha=0.08, zorder=0)
    return ax.axvspan(lower, upper, ymin=0.96, ymax=1.0, color=color, alpha=0.9)


def _diagnostic_title(fit_result):
    """Diagnostic title.

    Parameters
    ----------
    fit_result : lmfit.model.ModelResult
        Optimizer result with best-fit parameters.

    Returns
    -------
    str
        Title for the diagnostic plot.
    """
    kc = fit_result.params["kC"]
    sigma = fit_result.params["sigma"]
    kc_stderr = "unknown" if kc.stderr is None else f"{kc.stderr:.3g}"
    sigma_stderr = "unknown" if sigma.stderr is None else f"{sigma.stderr:.3g}"
    parameters = (
        f"kC={kc.value:.4g} ± {kc_stderr}; "
        f"reduced sigma={sigma.value:.4g} ± {sigma_stderr}"
    )
    return f"Spectrum fit diagnostic\n{parameters}"
