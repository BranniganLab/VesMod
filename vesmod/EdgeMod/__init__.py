"""Public core EdgeMod API."""

from .config import SpectrumFitConfig
from .fit_result import SpectrumFit
from .spectrum import Spectrum
from .spectrum_ensemble import SpectrumEnsemble

__all__ = [
    "SpectrumFitConfig",
    "SpectrumFit",
    "Spectrum",
    "SpectrumEnsemble",
]
