#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared data models for VesEdge.

Contains data structures representing detected contours, edge-extraction
results, and quality-control results.
"""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from vesmod.validation import (
    require_finite_array,
    require_finite_real,
    require_numeric_array,
)

from .qc.qc_config import VesicleQCConfig
from .qc.qc_result import QCResult


@dataclass(frozen=True)
class ImageContour:
    """
    A vesicle contour expressed in the coordinate system of an image.

    The radial coordinates are assumed to be sampled at evenly spaced angular
    positions spanning 0 to 2π, not including 2π itself. Successful
    construction guarantees a finite two-coordinate origin and a non-empty,
    one-dimensional, finite, positive real-valued radial array. The radial
    array is copied so mutation of the caller's input cannot invalidate the
    constructed contour.

    Attributes
    ----------
    origin : tuple[float, float]
        Cartesian coordinates (x, y) of the contour origin in image pixels.
    r : NDArray[np.float64]
        Radial distances from `origin` to the contour in image pixels.
    theta : NDArray[np.float64]
        Evenly spaced angular coordinates corresponding to `r`, ranging from
        0 to 2π, not including 2π.
    x : NDArray[np.float64]
        Cartesian x-coordinates of the contour. The first coordinate is
        repeated at the end to close the contour for plotting.
    y : NDArray[np.float64]
        Cartesian y-coordinates of the contour. The first coordinate is
        repeated at the end to close the contour for plotting.
    """

    origin: tuple[float, float]
    r: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and normalize contour coordinates."""
        if not isinstance(self.origin, tuple) or len(self.origin) != 2:
            raise TypeError("origin must be a two-coordinate tuple.")
        origin = tuple(
            require_finite_real(value, "origin coordinate")
            for value in self.origin
        )

        require_numeric_array(self.r, "r")
        if self.r.ndim != 1:
            raise ValueError("r must be a one-dimensional array.")
        if self.r.size == 0:
            raise ValueError("r must contain at least one radial sample.")
        if np.issubdtype(self.r.dtype, np.complexfloating):
            raise TypeError("r must contain real-valued numbers.")
        require_finite_array(self.r, "r")

        radii = np.asarray(self.r, dtype=float).copy()
        if np.any(radii <= 0):
            raise ValueError("r must contain only positive values.")

        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "r", radii)

    @property
    def theta(self) -> NDArray[np.float64]:
        """Return evenly spaced angular coordinates corresponding to `r`."""
        return np.linspace(
            0.0,
            2.0 * np.pi,
            self.r.shape[0],
            endpoint=False,
        )

    @property
    def x(self) -> NDArray[np.float64]:
        """Return x-coordinates, closing the contour for plotting."""
        x_vals = self.r * np.cos(self.theta) + self.origin[0]
        return np.append(x_vals, x_vals[0])

    @property
    def y(self) -> NDArray[np.float64]:
        """Return y-coordinates, closing the contour for plotting."""
        y_vals = self.r * np.sin(self.theta) + self.origin[1]
        return np.append(y_vals, y_vals[0])


@dataclass
class EdgeQC:
    """Quality-control flags and check-owned measurements for one detection.

    Each check records its registry name in `flags` when it rejects a frame
    and stores measurements in `diagnostics`. A detection passes when
    `flags` is empty.
    """

    flags: set[str] = field(default_factory=set)
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return whether the edge has passed all QC checks run so far."""
        return not self.flags


@dataclass
class EdgeDetection:
    """
    Successfully detected vesicle edge from one video frame.

    Attributes
    ----------
    full_contour : ImageContour
        Detected contour at the native angular sampling of the edge extractor.
    analysis_contour : ImageContour
        Contour used for subsequent analysis. If downsampling was requested,
        this is the downsampled contour. Otherwise, this is the same contour
        as `full_contour`.
    qc : EdgeQC
        Quality-control information associated with this detection.
    frame_index : int | None
        Zero-based source video frame index. None only for legacy/manually
        constructed results that have not yet been associated with a
        :class:`VesicleEdges` trajectory.
    """

    full_contour: ImageContour
    analysis_contour: ImageContour
    qc: EdgeQC = field(default_factory=EdgeQC)
    frame_index: int | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class EdgeDetectionFailure:
    """
    Record a frame for which edge extraction failed with an exception.

    Attributes
    ----------
    error : str
        Error message produced while attempting edge extraction.
    frame_index : int | None
        Zero-based source video frame index. None only for legacy/manually
        constructed results that have not yet been associated with a
        :class:`VesicleEdges` trajectory.
    """

    error: str
    frame_index: int | None = field(default=None, kw_only=True)


EdgeResult = EdgeDetection | EdgeDetectionFailure


@dataclass(frozen=True)
class VesicleQCResult:
    """Aggregate results from one completed VesEdge QC run.

    Attributes
    ----------
    config : VesicleQCConfig
        Configuration used for the QC run.
    results : Mapping[str, QCResult]
        Standard results keyed by the registered check name. Each check module
        declares the measurements it includes.
    frame_indices : tuple[int, ...]
        Source-frame indices of all successfully extracted contours evaluated.
    trajectory_flags : frozenset[str]
        Failures that apply to the complete video rather than individual
        detected frames.
    """

    config: VesicleQCConfig
    results: Mapping[str, QCResult] = field(default_factory=dict)
    frame_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate source-frame identity and the common result structure."""
        frame_indices = tuple(self.frame_indices)
        for name, result in self.results.items():
            if type(result) is not QCResult:
                raise TypeError("Every check must provide a QCResult.")
            if result.check_name != name:
                raise ValueError("QC result keys must match their check names.")
            if result.frame_indices != frame_indices:
                raise ValueError("All QC results must describe the aggregate source frames.")
        common = QCResult(
            check_name="aggregate", frame_indices=frame_indices,
            rejected=tuple(False for _ in frame_indices),
        )
        object.__setattr__(self, "frame_indices", common.frame_indices)
        object.__setattr__(self, "results", dict(self.results))

    def get_result(self, name: str) -> QCResult | None:
        """Return a registered check's result, if that check produced one."""
        return self.results.get(name)

    def to_dict(self) -> dict:
        """Serialize the configuration and standard check results."""
        return {
            "config": self.config.to_dict(),
            "results": {name: result.to_dict() for name, result in self.results.items()},
            "frame_indices": list(self.frame_indices),
            "trajectory_flags": sorted(self.trajectory_flags),
        }

    @classmethod
    def from_dict(cls, values: dict) -> "VesicleQCResult":
        """Load the current result format through registered check declarations."""
        from .qc.qc_registry import QC_CHECKS

        if not isinstance(values, dict) or set(values) != {"config", "results", "frame_indices", "trajectory_flags"}:
            raise ValueError("Invalid VesicleQCResult fields.")
        config = VesicleQCConfig.from_dict(values["config"])
        checks = {check.name: check for check in QC_CHECKS}
        saved_results = values["results"]
        expected = {name for name, check in checks.items() if check.enabled(config.get_config(name))}
        if not isinstance(saved_results, dict) or set(saved_results) != expected:
            raise ValueError("Saved QC results must match the enabled checks.")
        results = {
            name: checks[name].result_from_dict(data)
            for name, data in saved_results.items()
        }
        recorded = cls(
            config=config, results=results, frame_indices=tuple(values["frame_indices"]),
        )
        if not isinstance(values["trajectory_flags"], list):
            raise ValueError("Saved trajectory flags must be a list of strings.")
        if sorted(recorded.trajectory_flags) != values["trajectory_flags"]:
            raise ValueError("Saved trajectory flags do not match check results.")
        return recorded

    @property
    def trajectory_flags(self) -> frozenset[str]:
        """Collect recording-level rejection reasons from the check results."""
        return frozenset(
            flag for result in self.results.values() for flag in result.trajectory_flags
        )

    @property
    def accepted_frame_indices(self) -> tuple[int, ...]:
        """Return source frames accepted by every enabled check."""
        if self.trajectory_flags:
            return ()
        rejected = {
            index
            for result in self.results.values()
            for index, failed in zip(result.frame_indices, result.rejected, strict=True)
            if failed
        }
        return tuple(index for index in self.frame_indices if index not in rejected)

    @property
    def passed(self) -> bool:
        """Return whether the trajectory passed trajectory-level QC."""
        return not self.trajectory_flags
