"""Scientific configuration records for core EdgeMod spectrum fitting."""

from __future__ import annotations

from dataclasses import dataclass

from vesmod.validation import require_integer, require_positive_real, require_nonnegative_real


@dataclass(frozen=True)
class SpectrumFitConfig:
    """Configure one physical fit of a vesicle fluctuation spectrum.

    This configuration selects the equilibrium spectrum, optional camera
    averaging, and relaxation dynamics. The fitting interval is a fixed
    lower-inclusive, upper-exclusive q range. Experimental procedures may choose those bounds upstream, but the
    physical fitter does not depend on how they were selected.

    ``lmax`` is the inclusive upper spherical-mode summation bound. It must
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
    model : str, default="spherical_contour_spectrum"
        Equilibrium contour-power model (Hackl, Seifert, and Sackmann 1997).
    camera_integration : bool, default=False
        Apply finite-exposure averaging to each spherical-mode contribution.
        A positive exposure_time is required when enabled. Disabling this
        preserves exposure metadata without applying the correction.
    relaxation_model : str, default="solvent_only"
        "solvent_only" uses separate interior/exterior viscosities (Faizi 2020,
        Eq. 2). "solvent_and_membrane" uses a common solvent viscosity plus
        membrane drag (Faizi 2024, Eq. 1). Parameters never select the method.
    exposure_time : float, default=0.0
        Camera exposure duration in seconds.
    viscosity_in, viscosity_out : float or None
        Dynamic viscosities in Pa s. Camera integration requires both for
        solvent_only. For solvent_and_membrane, viscosity_out supplies the
        common solvent viscosity; viscosity_in may be omitted or equal it.
    eta_m : float or None, default=None
        Fixed membrane surface viscosity in Pa s m, required (including zero)
        for solvent_and_membrane and prohibited for solvent_only. Ensembles
        share eta_m; chi_s = eta_m/(R0*viscosity_out), with R0 in meters.
    """

    lmax: int = 500
    free_sigma: bool = True
    temperature: float = 295.0
    lower_bound: int = 3
    upper_bound: int = 8
    exposure_time: float = 0.0
    viscosity_in: float | None = None
    viscosity_out: float | None = None
    eta_m: float | None = None

    model: str = "spherical_contour_spectrum"
    camera_integration: bool = False
    relaxation_model: str = "solvent_only"

    def __post_init__(self) -> None:
        """Validate physical-fit parameters and establish config invariants."""
        for name in ("viscosity_in", "viscosity_out"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_positive_real(value, name))
        object.__setattr__(self, "exposure_time",
                           require_nonnegative_real(self.exposure_time, "exposure_time"))
        if self.model != "spherical_contour_spectrum":
            raise ValueError("model must be spherical_contour_spectrum.")
        if not isinstance(self.camera_integration, bool):
            raise TypeError("camera_integration must be a bool.")
        if self.relaxation_model not in {"solvent_only", "solvent_and_membrane"}:
            raise ValueError("relaxation_model must be solvent_only or solvent_and_membrane.")
        if self.eta_m is not None:
            object.__setattr__(self, "eta_m", require_nonnegative_real(self.eta_m, "eta_m"))
        if self.relaxation_model == "solvent_only":
            if self.eta_m is not None:
                raise ValueError("eta_m requires relaxation_model='solvent_and_membrane'.")
        else:
            if self.eta_m is None:
                raise ValueError("eta_m is required for solvent_and_membrane (zero is allowed).")
            if (self.viscosity_in is not None and self.viscosity_out is not None
                    and self.viscosity_in != self.viscosity_out):
                raise ValueError("Faizi 2024 Eq. 1 requires equal solvent viscosities; "
                                 "omit viscosity_in or set it equal to viscosity_out.")
        if self.camera_integration:
            if self.exposure_time <= 0:
                raise ValueError("Positive exposure_time is required with camera_integration=True.")
            if self.relaxation_model == "solvent_only":
                if self.viscosity_in is None or self.viscosity_out is None:
                    raise ValueError("viscosity_in and viscosity_out are required for solvent_only camera integration.")
            elif self.viscosity_out is None:
                raise ValueError("viscosity_out is required for solvent_and_membrane camera integration.")
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
            raise ValueError("lower_bound must be at least 2 for spherical_contour_spectrum.")
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound.")
        if lmax <= 0:
            raise ValueError("lmax must be positive.")
        if lmax < upper_bound - 1:
            raise ValueError(
                "lmax must be at least upper_bound - 1 because spherical_contour_spectrum uses "
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
            "exposure_time": self.exposure_time,
            "viscosity_in": self.viscosity_in,
            "viscosity_out": self.viscosity_out,
            "eta_m": self.eta_m,
            "model": self.model,
            "camera_integration": self.camera_integration,
            "relaxation_model": self.relaxation_model,
        }
