"""Immutable fit-result records for core EdgeMod spectrum analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Iterator

from vesmod import __version__
from vesmod.edgemod.fitting.config import SpectrumFitConfig


def _finite(value):
    """Convert unavailable or nonfinite optimizer diagnostics to None."""
    return float(value) if value is not None and isfinite(value) else None


@dataclass(frozen=True)
class FitDiagnostics:
    """Optimizer outcome and uncertainty in fitted parameter units.

    Covariance rows and columns follow parameter_names. Fixed parameters are
    omitted. Unavailable diagnostics are represented by None. Bounds are
    recorded as the lower/upper pairs for kC and reduced sigma, in that order.
    The optimizer success flag is independent of post-fit validation policy.
    """

    parameter_names: tuple[str, ...] = ()
    parameter_stderr: tuple[float | None, ...] = ()
    covariance: tuple[tuple[float | None, ...], ...] | None = None
    success: bool | None = None
    message: str | None = None
    method: str | None = None
    nfev: int | None = None
    chisqr: float | None = None
    redchi: float | None = None
    aic: float | None = None
    bic: float | None = None
    rmse: float | None = None
    relative_rmse: float | None = None
    post_fit_validation: str = "none"
    bounds: tuple[tuple[float | None, ...], ...] | None = None
    modes: tuple[int, ...] = ()
    weights: tuple[float, ...] | None = None

    @classmethod
    def from_optimizer(cls, result):
        """Copy diagnostics from an lmfit ModelResult into a durable record."""
        names = tuple(getattr(result, "var_names", ()) or ())
        covariance = getattr(result, "covar", None)
        bounds = getattr(result, "power_fit_bounds", None)
        success = getattr(result, "success", None)
        return cls(
            parameter_names=names,
            parameter_stderr=tuple(_finite(result.params[name].stderr) for name in names),
            covariance=None if covariance is None else tuple(
                tuple(_finite(value) for value in row) for row in covariance
            ),
            success=None if success is None else bool(success),
            message=getattr(result, "message", None),
            method=getattr(result, "method", None),
            nfev=getattr(result, "nfev", None),
            chisqr=_finite(getattr(result, "chisqr", None)),
            redchi=_finite(getattr(result, "redchi", None)) if getattr(result, "nfree", 1) > 0 else None,
            aic=_finite(getattr(result, "aic", None)),
            bic=_finite(getattr(result, "bic", None)),
            rmse=_finite(getattr(result, "power_rmse", None)),
            relative_rmse=_finite(getattr(result, "power_relative_rmse", None)),
            bounds=None if bounds is None else tuple(
                tuple(_finite(value) for value in pair) for pair in bounds
            ),
            modes=getattr(result, "power_fit_modes", ()),
            weights=getattr(result, "power_fit_weights", None),
        )


class _FitSerialization:
    """JSON serialization shared by individual and ensemble fit records."""

    def to_dict(self) -> dict:
        """Return fit parameters, scientific configuration, and diagnostics."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, data):
        """Restore a saved fit from a dict, including unversioned fit records."""
        data = dict(data)
        if data.get("schema_version", 1) != 1:
            raise ValueError("Unsupported fit result schema version.")
        data.setdefault("vesmod_version", None)
        data["config"] = SpectrumFitConfig(**data["config"])
        diagnostics = data.get("diagnostics")
        if diagnostics is not None:
            diagnostics = dict(diagnostics)
            for key in ("parameter_names", "parameter_stderr", "modes", "weights"):
                if diagnostics.get(key) is not None:
                    diagnostics[key] = tuple(diagnostics[key])
            for key in ("covariance", "bounds"):
                if diagnostics.get(key) is not None:
                    diagnostics[key] = tuple(tuple(row) for row in diagnostics[key])
            data["diagnostics"] = FitDiagnostics(**diagnostics)
        if data.get("radii") is not None:
            data["radii"] = tuple(data["radii"])
        return cls(**data)

    @classmethod
    def from_json(cls, path):
        """Restore one fit record from a JSON file path."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class SpectrumFit(_FitSerialization):
    """Store one physical fit of a :class:`Spectrum` without overwriting others.

    A ``Spectrum`` may be fit repeatedly over different q ranges. Each
    successful fit is represented by its own immutable ``SpectrumFit`` so the
    fitted values and exact physical-fit configuration remain associated.

    Experimental procedures that choose q bounds retain their own selection
    diagnostics separately from this core physical-fit result.

    Parameters
    ----------
    kC : float
        Fitted bending modulus.
    surface_tension : float
        Surface tension converted from the fitted reduced tension.
    lower_bound : int
        Inclusive lower q bound actually used for the physical fit.
    upper_bound : int
        Exclusive upper q bound actually used for the physical fit.
    config : SpectrumFitConfig
        Scientific configuration used to produce this physical fit.
    power_definition : str
        Definition of Fourier-mode power fitted to obtain these values.
    """

    kC: float
    surface_tension: float
    lower_bound: int
    upper_bound: int
    config: SpectrumFitConfig
    power_definition: str = "mean_square"

    reduced_sigma: float | None = None
    r0: float | None = None
    diagnostics: FitDiagnostics | None = None
    schema_version: int = 1
    vesmod_version: str | None = __version__

    def __iter__(self) -> Iterator[float]:
        """Allow compatibility unpacking as ``kc, tension = fit``."""
        yield self.kC
        yield self.surface_tension

@dataclass(frozen=True)
class EnsembleFit(_FitSerialization):
    """Store one fit of an averaged replica spectrum.

    An ensemble does not have one physical mean radius. Its fitted sigma is
    therefore retained as dimensionless reduced tension in the spherical
    contour model rather than converted to SI surface tension.
    """

    kC: float
    reduced_sigma: float
    config: SpectrumFitConfig
    weight_by_replica_sem: bool = False
    chisqr: float | None = None
    redchi: float | None = None
    radii: tuple[float, ...] | None = None

    diagnostics: FitDiagnostics | None = None
    replica_count: int | None = None
    replica_kC_std: float | None = None
    replica_kC_sem: float | None = None
    schema_version: int = 1
    vesmod_version: str | None = __version__
