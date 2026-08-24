"""Configuration records for EdgeMod spectrum fitting."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from numbers import Integral, Real

from .fit_range_selection import (
    FitRangeSelector,
    FixedFitRangeSelector,
)


@dataclass(frozen=True)
class SpectrumFitConfig:
    """Configure physical fitting of one vesicle fluctuation spectrum.

    Parameters
    ----------
    lmax : int
        Maximum summation index in the theoretical spectrum model.
    free_sigma : bool
        Whether reduced surface tension is fitted as a free parameter.
    temperature : float
        Experimental temperature in Kelvin.
    range_selector : FitRangeSelector
        Strategy used to select the lower-inclusive, upper-exclusive q range.
    """

    lmax: int = 500
    free_sigma: bool = True
    temperature: float = 295.0
    range_selector: FitRangeSelector = field(
        default_factory=lambda: FixedFitRangeSelector(
            lower_bound=3,
            upper_bound=8,
        )
    )

    def __post_init__(self) -> None:
        """Validate spectrum-fitting configuration."""
        if not isinstance(self.lmax, Integral) or isinstance(self.lmax, bool):
            raise TypeError("lmax must be an integer.")
        if self.lmax <= 0:
            raise ValueError("lmax must be positive.")
        if not isinstance(self.free_sigma, bool):
            raise TypeError("free_sigma must be a bool.")
        if not isinstance(self.temperature, Real) or isinstance(
            self.temperature,
            bool,
        ):
            raise TypeError("temperature must be numeric.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not hasattr(self.range_selector, "select"):
            raise TypeError("range_selector must provide a select method.")

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the fit configuration."""
        selector = self.range_selector
        selector_data = {"type": type(selector).__name__}
        selector_fields = getattr(selector, "__dataclass_fields__", None)
        if selector_fields is not None:
            for selector_field in fields(selector):
                selector_data[selector_field.name] = getattr(
                    selector,
                    selector_field.name,
                )

        return {
            "lmax": int(self.lmax),
            "free_sigma": self.free_sigma,
            "temperature": float(self.temperature),
            "range_selector": selector_data,
        }
