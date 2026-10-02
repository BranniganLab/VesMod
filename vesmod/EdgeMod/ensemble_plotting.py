"""Default ensemble spectrum figures, matching the DOPC/C16 references."""

from dataclasses import dataclass
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np

from .spectrum_utils import (
    HSS97,
    HSS97_with_camera_integration_time,
    calc_tension_from_reduced_tension,
)


@dataclass(frozen=True)
class EnsemblePlotConfig:
    """Shared axis limits and typography; no global matplotlib style changes.

    Limits default to the reference's q=2..20 view. Change them explicitly
    when displaying a different mode or power range. ``power_definition``
    describes the input data; it never transforms the powers.
    """

    xlim: tuple[float, float] = (1.8, 22.5)
    ylim: tuple[float, float] = (5e-8, 5e-3)
    power_definition: str = "mean_square"
    fontsize: float = 10

    def __post_init__(self):
        for name in ("xlim", "ylim"):
            limits = getattr(self, name)
            if len(limits) != 2 or not np.all(np.isfinite(limits)) or not 0 < limits[0] < limits[1]:
                raise ValueError(f"{name} must contain two increasing positive limits.")
        if self.power_definition not in {"mean_square", "temporal_variance"}:
            raise ValueError("power_definition must be mean_square or temporal_variance.")
        if not np.isfinite(self.fontsize) or self.fontsize <= 0:
            raise ValueError("fontsize must be finite and positive.")


def _latest_fits(ensemble, fits):
    """Select the latest recorded fit of each kind without running a fitter."""
    selected = {}
    for fit in ensemble.fit_results if fits is None else fits:
        selected[fit.config.free_sigma] = fit
    return [selected[key] for key in (False, True) if key in selected]


def _fit_prediction(fit, modes):
    """Evaluate the model and radius ensemble recorded by this fit."""
    if fit.config.exposure_time > 0:
        return HSS97_with_camera_integration_time(
            modes, fit.kC, fit.reduced_sigma, fit.config.lmax,
            config=fit.config, radii=fit.radii,
        )
    return HSS97(modes, fit.kC, fit.reduced_sigma, fit.config.lmax)


def _sigma_label(ensemble, fit):
    """Report SI sigma only when every replica's radius is available."""
    radii = fit.radii if fit.radii is not None else ensemble.radii_list
    if len(radii) == len(ensemble.spectra_list) and all(r is not None for r in radii):
        sigma = calc_tension_from_reduced_tension(
            float(np.mean(radii)), fit.reduced_sigma, fit.kC, fit.config.temperature,
        )
        return rf"$\sigma$={sigma:.3g} N/m (mean radius)"
    return rf"Reduced $\sigma$={fit.reduced_sigma:.3g}"


def plot_ensemble_spectrum(
    ensemble, *, label="Ensemble", ax=None, fits=None, config=None,
    legend=True, title_fit_values=False,
):
    """Plot replicas, equal-replica mean ± SEM, and recorded ensemble fits.

    Parameters
    ----------
    ensemble : SpectrumEnsemble
        Source replica spectra. At least one replica is required; SEM is
        omitted for a singleton because its sample SEM is undefined.
    label : str
        Composition/concentration text for the panel title.
    ax : matplotlib.axes.Axes, optional
        Existing axis. Otherwise create a single reference-sized figure.
    fits : sequence[EnsembleFit], optional
        Explicit fits; an empty sequence draws no fits. By default use the
        latest recorded fixed and free sigma fits, without refitting.
    config : EnsemblePlotConfig, optional
        Axis limits, font size, and actual input power definition.
    legend : bool
        Show the single-panel legend in the lower left.
    title_fit_values : bool
        Put fitted values in a multiline panel title (the C16 panel style).

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axis, ready for further editing or saving.
    """
    config = EnsemblePlotConfig() if config is None else config
    if not ensemble.spectra_list:
        raise ValueError("At least one replica spectrum is required.")
    modes = np.asarray(ensemble.modes)
    powers = np.asarray(ensemble.spectra_list, dtype=float)
    if powers.ndim != 2 or powers.shape[1] != modes.size:
        raise ValueError("Replica powers must match the ensemble modes.")
    mask = (modes >= 2) & (modes >= config.xlim[0]) & (modes <= config.xlim[1])
    order = np.argsort(modes[mask])
    q = modes[mask][order]
    powers = powers[:, mask][:, order]
    if q.size == 0:
        raise ValueError("No q >= 2 modes lie within the plotting limits.")
    if np.any(~np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("Displayed powers must be finite and nonnegative.")
    created_figure = ax is None
    if created_figure:
        _, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor("white")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(False, which="both")
    for replica in powers:
        ax.plot(q, np.where(replica > 0, replica, np.nan), "o-", color="0.6",
                alpha=0.4, linewidth=0.6, markersize=2.5, zorder=1)
    selected_fits = _latest_fits(ensemble, fits)
    title_values = []
    sigma_text = None
    for fit in selected_fits:
        selected = (q >= fit.config.lower_bound) & (q < fit.config.upper_bound)
        fit_q = q[selected]
        if not fit_q.size:
            raise ValueError("A supplied fit has no modes in the displayed range.")
        free = fit.config.free_sigma
        kind = "Free sigma" if free else "Fixed zero sigma"
        fit_label = f"{kind}: $k_C$={fit.kC:.2f} $k_BT$"
        if free:
            sigma_text = _sigma_label(ensemble, fit)
            fit_label += "\n" + sigma_text
        ax.plot(fit_q, _fit_prediction(fit, fit_q),
                color="#8c299b" if free else "#da7325",
                linestyle="--" if free else "-", linewidth=2,
                label=fit_label, zorder=3)
        title_values.append(f"{'free' if free else 'fixed'} {fit.kC:.2f}")
    mean = np.mean(powers, axis=0)
    sem = np.std(powers, axis=0, ddof=1) / np.sqrt(len(powers)) if len(powers) > 1 else None
    ax.errorbar(q, np.where(mean > 0, mean, np.nan), yerr=sem,
                fmt="o", color="#174c78", markersize=4, capsize=2.5,
                elinewidth=1.5, label="Equal-replica mean ± SEM" if sem is not None
                else "Equal-replica mean (SEM undefined; n=1)", zorder=4)
    ax.set_xlim(config.xlim)
    ax.set_ylim(config.ylim)
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", labelsize=config.fontsize)
    ax.set_xlabel("Fourier mode q", fontsize=config.fontsize)
    ylabel = "Temporal-variance power" if config.power_definition == "temporal_variance" else "Mean-square power"
    ax.set_ylabel(ylabel, fontsize=config.fontsize)
    title = f"{label}; n={len(powers)}" if title_fit_values else f"{label}: {len(powers)} selected replicas"
    if title_fit_values and title_values:
        title += "\n" + r"$k_C$: " + "; ".join(title_values) + r" $k_BT$"
        if sigma_text is not None:
            title += "\n" + sigma_text
    ax.set_title(title, fontsize=config.fontsize if title_fit_values else config.fontsize + 2)
    if legend:
        ax.legend(loc="lower left", frameon=False, fontsize=config.fontsize - 2)
    if created_figure:
        ax.figure.tight_layout()
    return ax.figure, ax


def plot_ensemble_spectra(ensembles, *, labels, fits=None, config=None, ncols=4, title=None):
    """Plot multiple ensembles with shared limits and one figure-level legend.

    ``labels`` and optional ``fits`` contain one entry per ensemble. Layout
    defaults to four columns (the C16 reference). No footer is created.
    Return ``(figure, axes)`` with axes always a two-dimensional array.
    """
    ensembles, labels = list(ensembles), list(labels)
    if not ensembles or len(labels) != len(ensembles):
        raise ValueError("Supply one label per ensemble and at least one ensemble.")
    if isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1:
        raise ValueError("ncols must be a positive integer.")
    fits = [None] * len(ensembles) if fits is None else list(fits)
    if len(fits) != len(ensembles):
        raise ValueError("Supply one fit sequence per ensemble.")
    ncols = min(ncols, len(ensembles))
    nrows = ceil(len(ensembles) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.7 * nrows + 0.7), squeeze=False)
    handles_by_kind = {}
    for ax, ensemble, label, panel_fits in zip(axes.flat, ensembles, labels, fits):
        plot_ensemble_spectrum(ensemble, label=label, ax=ax, fits=panel_fits,
                               config=config, legend=False, title_fit_values=True)
        handles, legend_labels = ax.get_legend_handles_labels()
        for handle, text in zip(handles, legend_labels):
            if text.startswith("Fixed"):
                handles_by_kind["Fixed-zero-sigma ensemble fit"] = handle
            elif text.startswith("Free"):
                handles_by_kind["Free-sigma ensemble fit"] = handle
            else:
                handles_by_kind[text] = handle
    for ax in list(axes.flat)[len(ensembles):]:
        ax.set_visible(False)
    if title:
        figure.suptitle(title, fontsize=16, y=0.99)
    figure.legend(list(handles_by_kind.values()), list(handles_by_kind),
                  loc="upper center", bbox_to_anchor=(0.5, 0.96 if title else 1),
                  ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.90 if title else 0.94))
    return figure, axes
