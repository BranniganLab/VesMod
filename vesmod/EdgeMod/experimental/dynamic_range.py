"""Experimental q^-3-based selection of EdgeMod fitting ranges."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterator
import math

import numpy as np


@dataclass(frozen=True)
class DynamicRangeSelection:
    """Result of experimental q^-3 range selection.

    Parameters
    ----------
    accepted : bool
        Whether the selected range satisfies the configured criteria.
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


@dataclass(frozen=True)
class QMinusThreeRangeSelector:
    """Select the longest trusted q range consistent with q^-3 scaling.

    This selector is experimental. It searches only inside the configured q
    interval, considers contiguous integer-q windows, and accepts a window only
    when both its fitted log-log slope and its residual to a fixed q^-3 model
    satisfy the supplied criteria.

    Parameters
    ----------
    lower_bound : int
        Inclusive lowest q value eligible for selection.
    upper_bound : int
        Exclusive highest q value eligible for selection.
    min_modes : int
        Minimum number of consecutive integer modes in a candidate window.
    slope_tolerance : float
        Finite, non-negative maximum absolute difference between fitted slope
        and -3.
    max_log_rmse : float
        Finite, non-negative maximum RMSE to the fixed q^-3 model in
        natural-log space.
    """

    lower_bound: int
    upper_bound: int
    min_modes: int
    slope_tolerance: float
    max_log_rmse: float

    def __post_init__(self) -> None:
        """Validate experimental selection parameters."""
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
        if not math.isfinite(self.slope_tolerance):
            raise ValueError("slope_tolerance must be finite.")
        if self.slope_tolerance < 0:
            raise ValueError("slope_tolerance must be non-negative.")
        if not math.isfinite(self.max_log_rmse):
            raise ValueError("max_log_rmse must be finite.")
        if self.max_log_rmse < 0:
            raise ValueError("max_log_rmse must be non-negative.")

    def select(
        self,
        modes: np.ndarray,
        avg_amps2: np.ndarray,
    ) -> DynamicRangeSelection:
        """Return the longest acceptable contiguous q^-3 scaling range."""
        eligible_modes, eligible_amps = self._eligible_spectrum(modes, avg_amps2)
        if eligible_modes.size < self.min_modes:
            return DynamicRangeSelection(
                accepted=False,
                lower_bound=None,
                upper_bound=None,
                reason=(
                    "Trusted q range contains fewer than "
                    f"{self.min_modes} usable modes."
                ),
            )

        best_accepted = None
        best_rejected = None
        found_candidate = False
        for candidate in self._iter_candidate_windows(
            eligible_modes,
            eligible_amps,
        ):
            found_candidate = True
            if self._is_accepted(candidate):
                if (
                    best_accepted is None
                    or self._accepted_key(candidate) > self._accepted_key(best_accepted)
                ):
                    best_accepted = candidate
            elif (
                best_rejected is None
                or self._rejected_key(candidate) < self._rejected_key(best_rejected)
            ):
                best_rejected = candidate

        if not found_candidate:
            return DynamicRangeSelection(
                accepted=False,
                lower_bound=None,
                upper_bound=None,
                reason="Trusted q range contains no sufficiently long contiguous window.",
            )
        if best_accepted is not None:
            return best_accepted

        return DynamicRangeSelection(
            accepted=False,
            lower_bound=best_rejected.lower_bound,
            upper_bound=best_rejected.upper_bound,
            slope=best_rejected.slope,
            log_rmse=best_rejected.log_rmse,
            reason="No trusted q range satisfied the q^-3 scaling criteria.",
        )

    def _is_accepted(self, candidate: DynamicRangeSelection) -> bool:
        """Return whether one evaluated window satisfies both criteria."""
        return (
            abs(candidate.slope + 3.0) <= self.slope_tolerance
            and candidate.log_rmse <= self.max_log_rmse
        )

    @staticmethod
    def _accepted_key(candidate: DynamicRangeSelection) -> tuple[float, ...]:
        """Return the ordering key for accepted windows."""
        return (
            candidate.upper_bound - candidate.lower_bound,
            -candidate.log_rmse,
            -abs(candidate.slope + 3.0),
        )

    @staticmethod
    def _rejected_key(candidate: DynamicRangeSelection) -> tuple[float, ...]:
        """Return the ordering key for rejected windows."""
        return (
            candidate.log_rmse,
            abs(candidate.slope + 3.0),
            -(candidate.upper_bound - candidate.lower_bound),
        )

    def _iter_candidate_windows(
        self,
        modes: np.ndarray,
        avg_amps2: np.ndarray,
    ) -> Iterator[DynamicRangeSelection]:
        """Yield windows while avoiding repeated full-window calculations."""
        run_starts = np.concatenate(
            ([0], np.flatnonzero(np.diff(modes) != 1) + 1)
        )
        run_stops = np.concatenate((run_starts[1:], [modes.size]))
        for run_start, run_stop in zip(run_starts, run_stops, strict=True):
            if run_stop - run_start < self.min_modes:
                continue
            yield from self._windows_from_contiguous_run(
                modes[run_start:run_stop],
                avg_amps2[run_start:run_stop],
            )

    def _windows_from_contiguous_run(
        self,
        modes: np.ndarray,
        avg_amps2: np.ndarray,
    ) -> Iterator[DynamicRangeSelection]:
        """Yield windows using stable constant-cost incremental statistics."""
        log_q = np.log(modes.astype(float))
        log_amp = np.log(avg_amps2)
        fixed_coordinate = log_amp + 3.0 * log_q

        for start in range(modes.size):
            count = 0
            mean_x = 0.0
            mean_y = 0.0
            mean_z = 0.0
            sum_xx = 0.0
            sum_xy = 0.0
            sum_zz = 0.0
            for stop in range(start, modes.size):
                count += 1

                delta_x = log_q[stop] - mean_x
                mean_x += delta_x / count
                delta_y = log_amp[stop] - mean_y
                mean_y += delta_y / count
                sum_xx += delta_x * (log_q[stop] - mean_x)
                sum_xy += delta_x * (log_amp[stop] - mean_y)

                delta_z = fixed_coordinate[stop] - mean_z
                mean_z += delta_z / count
                sum_zz += delta_z * (fixed_coordinate[stop] - mean_z)

                if count < self.min_modes:
                    continue

                slope = sum_xy / sum_xx
                log_rmse = math.sqrt(max(sum_zz / count, 0.0))
                yield DynamicRangeSelection(
                    accepted=True,
                    lower_bound=int(modes[start]),
                    upper_bound=int(modes[stop]) + 1,
                    slope=float(slope),
                    log_rmse=float(log_rmse),
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
