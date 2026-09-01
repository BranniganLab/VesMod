"""Scientific configuration records for core EdgeMod spectrum fitting."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import math


@dataclass(frozen=True)
class SpectrumFitConfig:
    """Configure one physical fit of a vesicle fluctuation spectrum.

    This core configuration contains only parameters required by the physical
    HSS97 fit. The fitting interval is a fixed lower-inclusive, upper-exclusive
    q range. Experimental procedures may choose those bounds upstream, but the
    physical fitter does not depend on how they were selected.

    ``lmax`` is the exclusive upper summation bound in HSS97. Therefore it must
    be at least ``upper_bound`` so the highest fitted mode contributes at least
    its ``l=q`` term.

    The historical positional order of ``lmax``, ``free_sigma``, and
    ``temperature`` is retained for compatibility. New q-bound fields follow
    those existing parameters.

    Parameters
    ----------
    lmax : int, default=500
        Exclusive upper summation bound in the theoretical spectrum model.
    free_sigma : bool, default=True
        Whether reduced surface tension is fitted as a free parameter.
    temperature : float, default=295.0
        Finite, positive experimental temperature in Kelvin used when
        converting reduced tension to surface tension.
    lower_bound : int, default=3
        Inclusive lowest Fourier mode used for the physical fit.
    upper_bound : int, default=8
        Exclusive upper Fourier-mode bound. The defaults therefore fit
        q = 3, 4, 5, 6, 7.
    """

    lmax: int = 500
    free_sigma: bool = True
    temperature: float = 295.0
    lower_bound: int = 3
    upper_bound: int = 8

    def __post_init__(self) -> None:
        """Validate physical-fit parameters."""
        for name, value in (
            ("lower_bound", self.lower_bound),
            ("upper_bound", self.upper_bound),
            ("lmax", self.lmax),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer.")

        if self.lower_bound <= 0:
            raise ValueError("lower_bound must be positive.")
        if self.lower_bound < 2:
            raise ValueError("lower_bound must be at least 2 for HSS97.")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound.")
        if self.lmax <= 0:
            raise ValueError("lmax must be positive.")
        if self.lmax < self.upper_bound:
            raise ValueError(
                "lmax must be at least upper_bound because HSS97 uses lmax "
                "as an exclusive summation bound."
            )
        if not isinstance(self.free_sigma, bool):
            raise TypeError("free_sigma must be a bool.")
        n_modes = self.upper_bound - self.lower_bound
        n_varying_parameters = 2 if self.free_sigma else 1
        if n_modes < n_varying_parameters:
            raise ValueError(
                "Configured q range contains too few modes for the number of "
                "varying fit parameters."
            )
        if not isinstance(self.temperature, Real) or isinstance(
            self.temperature,
            bool,
        ):
            raise TypeError("temperature must be numeric.")
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

    def to_dict(self) -> dict:
        """Return physical-fit settings in JSON-serializable form."""
        return {
            "lmax": int(self.lmax),
            "free_sigma": self.free_sigma,
            "temperature": float(self.temperature),
            "lower_bound": int(self.lower_bound),
            "upper_bound": int(self.upper_bound),
        }
