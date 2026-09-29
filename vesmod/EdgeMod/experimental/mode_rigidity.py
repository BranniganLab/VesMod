"""Experimental per-mode apparent bending-rigidity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import Boltzmann

from ..fit_result import SpectrumFit
from ..spectrum import Spectrum
from ..spectrum_utils import HSS97


@dataclass(frozen=True)
class ModeRigidityResult:
    """Apparent bending rigidity by Fourier mode for one fitted spectrum."""

    modes: np.ndarray
    apparent_kc: np.ndarray
    measured_power: np.ndarray
    fitted_kc: float
    reduced_sigma: float
    lower_bound: int
    upper_bound: int
    power_definition: str


def calculate_mode_rigidity(
    spectrum: Spectrum,
    fit: SpectrumFit,
    *,
    max_mode: int | None = None,
) -> ModeRigidityResult:
    """Calculate apparent kC at each mode using one fit's shared tension.

    The selected power definition is used both by the physical fit and by
    this inversion. A flat result in the fitted interval is expected for data
    that follow HSS97 exactly.
    """
    if not isinstance(spectrum, Spectrum):
        raise TypeError("spectrum must be a Spectrum.")
    if not isinstance(fit, SpectrumFit):
        raise TypeError("fit must be a SpectrumFit.")
    power_definition = getattr(spectrum, "power_definition", "mean_square")
    if fit.power_definition != power_definition:
        raise ValueError("fit and spectrum use different power definitions.")
    if max_mode is not None:
        if isinstance(max_mode, bool) or not isinstance(max_mode, Integral):
            raise TypeError("max_mode must be an integer or None.")
        if max_mode < 2:
            raise ValueError("max_mode must be at least 2.")

    modes_all = np.asarray(spectrum.modes)
    power_all = np.asarray(spectrum.avg_amps2, dtype=float)
    if modes_all.shape != power_all.shape:
        raise ValueError("spectrum modes and powers must have matching shapes.")
    upper_mode = fit.config.lmax
    if max_mode is not None:
        upper_mode = min(upper_mode, int(max_mode))
    mask = (modes_all >= 2) & (modes_all <= upper_mode)
    modes = modes_all[mask].astype(int)
    measured_power = power_all[mask]
    if modes.size == 0:
        raise ValueError("Spectrum contains no modes in the requested range.")

    radius_m = float(spectrum.r0) / 1e6
    reduced_sigma = (
        fit.surface_tension * radius_m**2
        / (fit.kC * Boltzmann * fit.config.temperature)
    )
    unit_kc_power = np.asarray(
        HSS97(modes.tolist(), 1.0, reduced_sigma, fit.config.lmax),
        dtype=float,
    )
    apparent_kc = np.full(measured_power.shape, np.nan, dtype=float)
    valid = np.isfinite(measured_power) & (measured_power > 0)
    apparent_kc[valid] = unit_kc_power[valid] / measured_power[valid]

    return ModeRigidityResult(
        modes=modes,
        apparent_kc=apparent_kc,
        measured_power=measured_power,
        fitted_kc=float(fit.kC),
        reduced_sigma=float(reduced_sigma),
        lower_bound=fit.lower_bound,
        upper_bound=fit.upper_bound,
        power_definition=power_definition,
    )


def plot_mode_rigidity(
    result: ModeRigidityResult,
    *,
    ax=None,
    color=None,
    label=None,
):
    """Plot a mode-rigidity result on an existing or new axis."""
    if not isinstance(result, ModeRigidityResult):
        raise TypeError("result must be a ModeRigidityResult.")
    if ax is None:
        _, ax = plt.subplots()

    selected = (result.modes >= result.lower_bound) & (
        result.modes < result.upper_bound
    )
    finite = np.isfinite(result.apparent_kc) & (result.apparent_kc > 0)
    outside_fit = finite & ~selected
    inside_fit = finite & selected
    if np.any(outside_fit):
        line, = ax.plot(
            result.modes[outside_fit],
            result.apparent_kc[outside_fit],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=color,
            color=color,
            alpha=0.55,
        )
        color = line.get_color()
    if np.any(inside_fit):
        line, = ax.plot(
            result.modes[inside_fit],
            result.apparent_kc[inside_fit],
            linestyle="none",
            marker="o",
            color=color,
            label=label,
        )
        color = line.get_color()
    ax.axhline(
        result.fitted_kc,
        color=color if color is not None else "C0",
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_xlabel("Fourier mode q")
    ax.set_ylabel(r"Apparent $k_C(q)$ ($k_BT$)")
    return ax.figure, ax
