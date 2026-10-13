"""Public core EdgeMod API."""

from .spectrum_utils import spherical_contour_spectrum, predict_spectrum, mode_relaxation_times
from .config import SpectrumFitConfig
from .fit_result import EnsembleFit, SpectrumFit
from .spectrum import Spectrum
from .spectrum_ensemble import SpectrumEnsemble
from .spectrum_plotting import (
    SpectrumPlotConfig,
    SpectrumPlotData,
    SpectrumPlotResult,
    plot_q3_scaled_spectrum,
    plot_spectrum,
    save_spectrum_fit_diagnostic,
)

__all__ = [
    "spherical_contour_spectrum",
    "predict_spectrum",
    "mode_relaxation_times",
    "SpectrumFitConfig",
    "SpectrumFit",
    "EnsembleFit",
    "Spectrum",
    "SpectrumEnsemble",
    "SpectrumPlotConfig",
    "SpectrumPlotData",
    "SpectrumPlotResult",
    "plot_q3_scaled_spectrum",
    "plot_spectrum",
    "save_spectrum_fit_diagnostic",
]
