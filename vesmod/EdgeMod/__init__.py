"""Public core EdgeMod API."""

from .config import SpectrumFitConfig
from .fit_result import SpectrumFit
from .spectrum import Spectrum
from .spectrum_ensemble import SpectrumEnsemble
from .spectrum_plotting import (
    SpectrumPlotConfig,
    SpectrumPlotData,
    SpectrumPlotResult,
    plot_q4_scaled_spectrum,
    plot_spectrum,
    save_spectrum_fit_diagnostic,
)

__all__ = [
    "SpectrumFitConfig",
    "SpectrumFit",
    "Spectrum",
    "SpectrumEnsemble",
    "SpectrumPlotConfig",
    "SpectrumPlotData",
    "SpectrumPlotResult",
    "plot_q4_scaled_spectrum",
    "plot_spectrum",
    "save_spectrum_fit_diagnostic",
]
