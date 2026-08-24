"""Select trustworthy q ranges for EdgeMod spectrum fitting."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FitRangeSelection:
    """Result of evaluating a candidate q range for spectrum fitting.

    Parameters
    ----------
    accepted : bool
        Whether the selected range satisfies the selector criteria.
    lower_bound : int | None
        Inclusive lower q bound of the selected or best rejected range.
    upper_bound : int | None
        Exclusive upper q bound of the selected or best rejected range.
    slope : float | None
        Best-fit log-log power-law slope for the reported range.
    log_rmse : float | None
        Root-mean-square residual to a fixed q^-3 model in log space.
    reason : str | None
        Explanation when no acceptable range is found.
    """

    accepted: bool
    lower_bound: int | None
    upper_bound: int | None
    slope: float | None = None
    log_rmse: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        """Return JSON-serializable selection diagnostics."""
        return {
            "accepted": self.accepted,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "slope": self.slope,
            "log_rmse": self.log_rmse,
            "reason": self.reason,
        }


class FitRangeSelector(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for strategies that select q ranges from a spectrum."""

    def select(
        self,
        modes: np.ndarray,
        avg_amps2: np.ndarray,
    ) -> FitRangeSelection:
        """Return the q range that should be used for physical fitting."""


@dataclass(frozen=True)
class FixedFitRangeSelector:
    """Select a fixed lower-inclusive, upper-exclusive q range."""

    lower_bound: int
    upper_bound: int

    def select(
        self,
        modes: np.ndarray,
        _avg_amps2: np.ndarray,
    ) -> FitRangeSelection:
        """Return the configured range after validating that it is populated."""
        mask = (modes >= self.lower_bound) & (modes < self.upper_bound)
        if not np.any(mask):
            return FitRangeSelection(
                accepted=False,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                reason="Configured q range contains no spectrum modes.",
            )
        return FitRangeSelection(
            accepted=True,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )


@dataclass(frozen=True)
class QMinusThreeFitRangeSelector:
    """Select the longest trustworthy range consistent with q^-3 scaling.

    Candidate windows are restricted to the configured q interval, evaluated
    in log space, and required to contain consecutive integer modes. A window
    is accepted when both its unconstrained power-law slope is sufficiently
    close to -3 and its residual to a fixed q^-3 model is sufficiently small.

    Parameters
    ----------
    lower_bound : int
        Inclusive lowest q value eligible for selection.
    upper_bound : int
        Exclusive highest q value eligible for selection.
    min_modes : int
        Minimum number of consecutive integer modes in a candidate window.
    slope_tolerance : float
        Maximum allowed absolute difference between fitted slope and -3.
    max_log_rmse : float
        Maximum allowed RMSE to the fixed q^-3 model in natural-log space.
    """

    lower_bound: int
    upper_bound: int
    min_modes: int
    slope_tolerance: float
    max_log_rmse: float

    def __post_init__(self) -> None:
        """Validate selector configuration."""
        if not isinstance(self.lower_bound, Integral) or isinstance(
            self.lower_bound,
            bool,
        ):
            raise TypeError("lower_bound must be an integer q value.")
        if not isinstance(self.upper_bound, Integral) or isinstance(
            self.upper_bound,
            bool,
        ):
            raise TypeError("upper_bound must be an integer q value.")
        if not isinstance(self.min_modes, Integral) or isinstance(
            self.min_modes,
            bool,
        ):
            raise TypeError("min_modes must be an integer.")
        if not isinstance(self.slope_tolerance, Real):
            raise TypeError("slope_tolerance must be numeric.")
        if not isinstance(self.max_log_rmse, Real):
            raise TypeError("max_log_rmse must be numeric.")

        if self.lower_bound <= 0:
            raise ValueError("lower_bound must be a positive integer q value.")
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound.")
        if self.min_modes < 2:
            raise ValueError("min_modes must be at least 2.")
        if self.slope_tolerance < 0:
            raise ValueError("slope_tolerance must be non-negative.")
        if self.max_log_rmse < 0:
            raise ValueError("max_log_rmse must be non-negative.")

    def select(
        self,
        modes: np.ndarray,
        avg_amps2: np.ndarray,
    ) -> FitRangeSelection:
        """Return the longest acceptable contiguous q^-3 scaling range."""
        eligible_modes, eligible_amps = self._eligible_spectrum(modes, avg_amps2)
        if eligible_modes.size < self.min_modes:
            return FitRangeSelection(
                accepted=False,
                lower_bound=None,
                upper_bound=None,
                reason=(
                    "Trusted q range contains fewer than "
                    f"{self.min_modes} usable modes."
                ),
            )

        candidates = []
        for start in range(eligible_modes.size):
            for stop in range(start + self.min_modes, eligible_modes.size + 1):
                window_modes = eligible_modes[start:stop]
                if not np.all(np.diff(window_modes) == 1):
                    continue
                window_amps = eligible_amps[start:stop]
                candidates.append(
                    self._evaluate_window(window_modes, window_amps)
                )

        if not candidates:
            return FitRangeSelection(
                accepted=False,
                lower_bound=None,
                upper_bound=None,
                reason="Trusted q range contains no sufficiently long contiguous window.",
            )

        accepted = [
            candidate
            for candidate in candidates
            if (
                abs(candidate.slope + 3.0) <= self.slope_tolerance
                and candidate.log_rmse <= self.max_log_rmse
            )
        ]
        if accepted:
            return max(
                accepted,
                key=lambda candidate: (
                    candidate.upper_bound - candidate.lower_bound,
                    -candidate.log_rmse,
                    -abs(candidate.slope + 3.0),
                ),
            )

        best_rejected = min(
            candidates,
            key=lambda candidate: (
                candidate.log_rmse,
                abs(candidate.slope + 3.0),
                -(candidate.upper_bound - candidate.lower_bound),
            ),
        )
        return FitRangeSelection(
            accepted=False,
            lower_bound=best_rejected.lower_bound,
            upper_bound=best_rejected.upper_bound,
            slope=best_rejected.slope,
            log_rmse=best_rejected.log_rmse,
            reason="No trusted q range satisfied the q^-3 scaling criteria.",
        )

    def _eligible_spectrum(
        self,
        modes: np.ndarray,
        avg_amps2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return finite positive amplitudes inside the trusted q range."""
        modes = np.asarray(modes)
        avg_amps2 = np.asarray(avg_amps2)
        if modes.shape != avg_amps2.shape:
            raise ValueError("modes and avg_amps2 must have matching shapes.")

        mask = (
            (modes >= self.lower_bound)
            & (modes < self.upper_bound)
            & np.isfinite(avg_amps2)
            & (avg_amps2 > 0)
        )
        eligible_modes = modes[mask].astype(int, copy=False)
        eligible_amps = avg_amps2[mask].astype(float, copy=False)
        order = np.argsort(eligible_modes)
        return eligible_modes[order], eligible_amps[order]

    @staticmethod
    def _evaluate_window(
        modes: np.ndarray,
        avg_amps2: np.ndarray,
    ) -> FitRangeSelection:
        """Measure power-law slope and q^-3 residual for one candidate window."""
        log_q = np.log(modes.astype(float))
        log_amp = np.log(avg_amps2)
        slope, _ = np.polyfit(log_q, log_amp, 1)

        log_amplitude = float(np.mean(log_amp + 3.0 * log_q))
        fixed_model = log_amplitude - 3.0 * log_q
        log_rmse = float(np.sqrt(np.mean((log_amp - fixed_model) ** 2)))

        return FitRangeSelection(
            accepted=True,
            lower_bound=int(modes[0]),
            upper_bound=int(modes[-1]) + 1,
            slope=float(slope),
            log_rmse=log_rmse,
        )
