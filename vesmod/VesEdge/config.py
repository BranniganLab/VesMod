#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration models for VesEdge extraction and quality control."""

from dataclasses import dataclass

from vesmod.validation import (
    require_fraction,
    require_integer_valued,
    require_nonnegative_real,
    require_positive_real,
)


_CALIBRATION_SOURCES = {"measured", "assumed", "unspecified"}


@dataclass(frozen=True)
class EdgeExtractionConfig:
    """Configuration parameters for edge extraction and contour preparation.

    Successful construction guarantees a finite positive spatial calibration
    and either no angular downsampling or a positive integer sample count.
    Numeric values are normalized to ordinary Python ``float``/``int`` values,
    so downstream extraction code may rely on those invariants.

    Attributes
    ----------
    pixels_per_micron : float
        How many pixels in the image represent one micron in real space.
    n_angular_samples : int | None
        How many angular samples to downsample to. If None, do not downsample.
    calibration_source : {"measured", "assumed", "unspecified"}
        Provenance for the spatial calibration. ``"measured"`` means the value
        came from microscope calibration, ``"assumed"`` means unit calibration
        was explicitly requested, and ``"unspecified"`` preserves compatibility
        with older API calls and checkpoints that did not record this choice.
    """

    pixels_per_micron: float = 1
    n_angular_samples: int | None = 120
    calibration_source: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate and normalize edge-extraction configuration."""
        pixels_per_micron = require_positive_real(
            self.pixels_per_micron,
            "pixels_per_micron",
        )
        object.__setattr__(self, "pixels_per_micron", pixels_per_micron)

        if self.calibration_source not in _CALIBRATION_SOURCES:
            allowed = ", ".join(sorted(_CALIBRATION_SOURCES))
            raise ValueError(
                f"calibration_source must be one of: {allowed}."
            )

        if self.n_angular_samples is None:
            return

        n_angular_samples = require_integer_valued(
            self.n_angular_samples,
            "n_angular_samples",
        )
        if n_angular_samples <= 0:
            raise ValueError("n_angular_samples must be positive.")
        object.__setattr__(self, "n_angular_samples", n_angular_samples)


@dataclass(frozen=True)
class EdgeQCConfig:
    """Configuration parameters for VesEdge quality-control checks.

    Successful construction guarantees finite numeric QC thresholds within
    their allowed ranges and boolean enable/disable flags. QC execution code
    may therefore treat this object as validated configuration.

    Attributes
    ----------
    curvature_threshold : float
        Maximum allowed absolute wrapped finite second difference of a
        median-radius-normalized analysis contour. This threshold is
        dimensionless.
    enable_curvature_qc : bool
        Whether frame-level curvature QC runs. Default is True.
    max_relative_area_deviation : float
        Maximum allowed absolute fractional deviation from the trajectory
        median contour area. Default is 0.25.
    enable_area_qc : bool
        Whether trajectory-level contour-area QC runs. Default is True.
    """

    curvature_threshold: float
    enable_curvature_qc: bool = True
    max_relative_area_deviation: float = 0.25
    enable_area_qc: bool = True

    def __post_init__(self) -> None:
        """Validate quality-control configuration parameters.

        Raises
        ------
        TypeError
            If an enable/disable flag is not boolean or a numeric threshold is
            not a real number.
        ValueError
            If any numeric parameter lies outside its allowed range.
        """
        curvature_threshold = require_nonnegative_real(
            self.curvature_threshold,
            "curvature_threshold",
        )
        max_relative_area_deviation = require_fraction(
            self.max_relative_area_deviation,
            "max_relative_area_deviation",
            include_one=False,
        )

        if not isinstance(self.enable_curvature_qc, bool):
            raise TypeError("enable_curvature_qc must be a bool.")
        if not isinstance(self.enable_area_qc, bool):
            raise TypeError("enable_area_qc must be a bool.")

        object.__setattr__(self, "curvature_threshold", curvature_threshold)
        object.__setattr__(
            self,
            "max_relative_area_deviation",
            max_relative_area_deviation,
        )