"""Scientific configuration records for core EdgeMod spectrum fitting."""

from __future__ import annotations

from dataclasses import dataclass

from vesmod.validation import require_integer, require_positive_real


@dataclass(frozen=True)
class SpectrumFitConfig:
    """Configure one physical fit of a vesicle fluctuation spectrum.

    This core configuration contains only parameters required by the physical
    HSS97 fit. The fitting interval is a fixed lower-inclusive, upper-exclusive
    q range. Experimental procedures may choose those bounds upstream, but the
    physical fitter does not depend on how they were selected.

    ``lmax`` is the inclusive upper summation bound in HSS97. Therefore it must
    be at least ``upper_bound - 1`` so the highest fitted mode contributes at
    least its ``l=q`` term.

    Successful construction guarantees integer q/lmax bounds, a finite positive
    temperature, a valid fixed q interval, and enough fitted modes for the
    configured number of varying physical parameters. Downstream fit code may
    rely on those invariants rather than revalidating the same configuration.

    The historical positional order of ``lmax``, ``free_sigma``, and
    ``temperature`` is retained for compatibility. New q-bound fields follow
    those existing parameters.

    Parameters
    ----------
    lmax : int, default=500
        Inclusive upper summation bound in the theoretical spectrum model.
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
        """Validate physical-fit parameters and establish config invariants."""
        lower_bound = require_integer(self.lower_bound, "lower_bound")
        upper_bound = require_integer(self.upper_bound, "upper_bound")
        lmax = require_integer(self.lmax, "lmax")
        temperature = require_positive_real(self.temperature, "temperature")

        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "upper_bound", upper_bound)
        object.__setattr__(self, "lmax", lmax)
        object.__setattr__(self, "temperature", temperature)

        if lower_bound <= 0:
            raise ValueError("lower_bound must be positive.")
        if lower_bound < 2:
            raise ValueError("lower_bound must be at least 2 for HSS97.")
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound.")
        if lmax <= 0:
            raise ValueError("lmax must be positive.")
        if lmax < upper_bound - 1:
            raise ValueError(
                "lmax must be at least upper_bound - 1 because HSS97 uses "
                "lmax as an inclusive summation bound."
            )
        if not isinstance(self.free_sigma, bool):
            raise TypeError("free_sigma must be a bool.")
        n_modes = upper_bound - lower_bound
        n_varying_parameters = 2 if self.free_sigma else 1
        if n_modes < n_varying_parameters:
            raise ValueError(
                "Configured q range contains too few modes for the number of "
                "varying fit parameters."
            )

    def to_dict(self) -> dict:
        """Return physical-fit settings in JSON-serializable form."""
        return {
            "lmax": int(self.lmax),
            "free_sigma": self.free_sigma,
            "temperature": float(self.temperature),
            "lower_bound": int(self.lower_bound),
            "upper_bound": int(self.upper_bound),
        }
