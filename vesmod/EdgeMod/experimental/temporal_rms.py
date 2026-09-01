"""Experimental temporal-RMS screening of vesicle contour trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vesmod.validation import require_integer, require_nonnegative_real


@dataclass(frozen=True)
class TemporalRMSConfig:
    """Configure experimental absolute temporal-RMS screening.

    ``upper_bound`` is exclusive. When ``cutoff_nm`` is omitted, amplitudes are
    measured and reported without excluding any trajectory.

    Successful construction guarantees integer q bounds with a non-empty
    interval and, when present, a finite non-negative cutoff. The temporal-RMS
    calculation can therefore trust configuration invariants and focus its
    validation on the independently supplied trajectory and available modes.
    """

    lower_bound: int = 3
    upper_bound: int = 8
    cutoff_nm: float | None = None

    def __post_init__(self) -> None:
        """Validate experimental temporal-RMS parameters."""
        lower_bound = require_integer(self.lower_bound, "lower_bound")
        upper_bound = require_integer(self.upper_bound, "upper_bound")
        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "upper_bound", upper_bound)

        if lower_bound < 1:
            raise ValueError("lower_bound must be at least 1.")
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound.")

        if self.cutoff_nm is None:
            return
        cutoff_nm = require_nonnegative_real(self.cutoff_nm, "cutoff_nm")
        object.__setattr__(self, "cutoff_nm", cutoff_nm)

    def to_dict(self) -> dict:
        """Return JSON-serializable configuration values."""
        return {
            "lower_bound": int(self.lower_bound),
            "upper_bound": int(self.upper_bound),
            "cutoff_nm": (
                None if self.cutoff_nm is None else float(self.cutoff_nm)
            ),
        }


@dataclass(frozen=True)
class TemporalRMSResult:
    """Record one trajectory's experimental temporal-RMS decision."""

    amplitude_nm: float
    included: bool
    config: TemporalRMSConfig

    def to_dict(self) -> dict:
        """Return JSON-serializable amplitude, decision, and configuration."""
        return {
            "amplitude_nm": self.amplitude_nm,
            "included": self.included,
            "config": self.config.to_dict(),
        }


def calculate_temporal_rms(
    radii_microns: NDArray[np.floating],
    config: TemporalRMSConfig | None = None,
) -> TemporalRMSResult:
    """Calculate combined temporal RMS motion over selected Fourier modes.

    Each mode's temporal mean is removed so persistent noncircularity does not
    count as motion. The factor of two includes the conjugate negative-q modes
    of each real-valued contour.
    """
    if config is None:
        config = TemporalRMSConfig()
    if not isinstance(config, TemporalRMSConfig):
        raise TypeError("config must be a TemporalRMSConfig or None.")

    radii = np.asarray(radii_microns)
    if radii.ndim != 2:
        raise ValueError("radii_microns must be a two-dimensional array.")
    if radii.shape[0] == 0 or radii.shape[1] == 0:
        raise ValueError("radii_microns must contain frames and angular samples.")
    if not np.issubdtype(radii.dtype, np.number):
        raise TypeError("radii_microns must contain numeric values.")
    if not np.all(np.isfinite(radii)):
        raise ValueError("radii_microns must contain only finite values.")

    highest_exclusive_bound = (radii.shape[1] + 1) // 2
    if config.upper_bound > highest_exclusive_bound:
        raise ValueError("Requested temporal-RMS modes are not available.")

    amplitudes = np.fft.fft(radii, axis=1) / radii.shape[1]
    selected = amplitudes[:, config.lower_bound:config.upper_bound]
    temporal_deviations = selected - np.mean(selected, axis=0)
    mode_variances = np.mean(np.abs(temporal_deviations) ** 2, axis=0)
    amplitude_nm = float(1000.0 * np.sqrt(2.0 * np.sum(mode_variances)))
    included = config.cutoff_nm is None or amplitude_nm >= config.cutoff_nm
    return TemporalRMSResult(
        amplitude_nm=amplitude_nm,
        included=included,
        config=config,
    )
