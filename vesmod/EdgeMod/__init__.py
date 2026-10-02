"""Public core EdgeMod API."""

from .config import SpectrumFitConfig
from .fit_result import EnsembleFit, SpectrumFit
from .spectrum import Spectrum
from .spectrum_ensemble import SpectrumEnsemble
from .ensemble_plotting import (
    EnsemblePlotConfig,
    plot_ensemble_spectrum,
    plot_ensemble_spectra,
)
from .spectrum_plotting import (
    SpectrumPlotConfig,
    SpectrumPlotData,
    SpectrumPlotResult,
    plot_q3_scaled_spectrum,
    plot_spectrum,
    save_spectrum_fit_diagnostic,
)

__all__ = [
    "SpectrumFitConfig",
    "SpectrumFit",
    "EnsembleFit",
    "Spectrum",
    "SpectrumEnsemble",
    "EnsemblePlotConfig",
    "plot_ensemble_spectrum",
    "plot_ensemble_spectra",
    "SpectrumPlotConfig",
    "SpectrumPlotData",
    "SpectrumPlotResult",
    "plot_q3_scaled_spectrum",
    "plot_spectrum",
    "save_spectrum_fit_diagnostic",
]
