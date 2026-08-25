#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration models for VesEdge extraction and quality control."""

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class EdgeExtractionConfig:
    """Configuration parameters for edge extraction and contour preparation.

    Attributes
    ----------
    pixels_per_micron : float
        How many pixels in the image represent one micron in real space.
    n_angular_samples : int | None
        How many angular samples to downsample to. If None, do not downsample.
    """

    pixels_per_micron: float = 1
    n_angular_samples: int | None = 120

    def __post_init__(self) -> None:
        """Validate and normalize edge-extraction configuration."""
        if isinstance(self.pixels_per_micron, bool) or not isinstance(
            self.pixels_per_micron,
            Real,
        ):
            raise TypeError("pixels_per_micron must be a real number.")

        pixels_per_micron = float(self.pixels_per_micron)
        if not np.isfinite(pixels_per_micron):
            raise ValueError("pixels_per_micron must be finite.")
        if pixels_per_micron <= 0:
            raise ValueError("pixels_per_micron must be positive.")
        object.__setattr__(self, "pixels_per_micron", pixels_per_micron)

        if self.n_angular_samples is None:
            return
        if isinstance(self.n_angular_samples, bool) or not isinstance(
            self.n_angular_samples,
            Real,
        ):
            raise TypeError(
                "n_angular_samples must be an integer-valued number or None."
            )
        if not np.isfinite(self.n_angular_samples):
            raise ValueError("n_angular_samples must be finite.")
        if not float(self.n_angular_samples).is_integer():
            raise ValueError("n_angular_samples must be integer-valued.")

        n_angular_samples = int(self.n_angular_samples)
        if n_angular_samples <= 0:
            raise ValueError("n_angular_samples must be positive.")
        object.__setattr__(self, "n_angular_samples", n_angular_samples)


@dataclass(frozen=True)
class EdgeQCConfig:
    """Configuration parameters for VesEdge quality-control checks.

    Attributes
    ----------
    curvature_threshold : float
        Maximum allowed absolute wrapped finite second difference of an
        analysis contour.
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
        ValueError
            If any numeric parameter lies outside its allowed range.
        """
        if not np.isfinite(self.curvature_threshold):
            raise ValueError("curvature_threshold must be finite.")
        if self.curvature_threshold < 0:
            raise ValueError("curvature_threshold must be non-negative.")
        if not np.isfinite(self.max_relative_area_deviation):
            raise ValueError(
                "max_relative_area_deviation must be finite."
            )
        if not 0 <= self.max_relative_area_deviation < 1:
            raise ValueError(
                "max_relative_area_deviation must be at least 0 and less than 1."
            )

