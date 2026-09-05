#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration models for VesEdge extraction and quality control."""

from dataclasses import dataclass, field

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
class CurvatureQCConfig:
    """Configuration for frame-level curvature quality control."""

    threshold: float
    enabled: bool = True

    def __post_init__(self) -> None:
        threshold = require_nonnegative_real(self.threshold, "threshold")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        object.__setattr__(self, "threshold", threshold)


@dataclass(frozen=True)
class AreaQCConfig:
    """Configuration for trajectory-level contour-area quality control."""

    max_relative_deviation: float = 0.25
    enabled: bool = True

    def __post_init__(self) -> None:
        deviation = require_fraction(
            self.max_relative_deviation,
            "max_relative_deviation",
            include_one=False,
        )
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        object.__setattr__(self, "max_relative_deviation", deviation)


@dataclass(frozen=True)
class InternalVesicleQCConfig:
    """Configuration for experimental internal-vesicle selection QC."""

    enabled: bool = False
    max_area_fraction: float = 0.5
    min_radius_ratio: float = 1.15
    min_separation_pixels: float = 5.0
    gradient_ratio: float = 0.5
    max_radial_deviation_fraction: float = 0.15
    min_angular_coverage: float = 0.6
    max_frames: int = 20
    min_valid_frames: int = 3
    min_valid_frame_fraction: float = 0.5
    min_frame_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Validate and normalize internal-vesicle thresholds."""
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        fractions = (
            "max_area_fraction",
            "max_radial_deviation_fraction",
            "min_angular_coverage",
            "min_valid_frame_fraction",
            "min_frame_fraction",
        )
        for name in fractions:
            object.__setattr__(self, name, require_fraction(getattr(self, name), name))
        min_radius_ratio = require_positive_real(
            self.min_radius_ratio,
            "min_radius_ratio",
        )
        if min_radius_ratio <= 1:
            raise ValueError("min_radius_ratio must be greater than 1.")
        object.__setattr__(self, "min_radius_ratio", min_radius_ratio)
        for name in ("min_separation_pixels", "gradient_ratio"):
            object.__setattr__(
                self,
                name,
                require_nonnegative_real(getattr(self, name), name),
            )
        for name in ("max_frames", "min_valid_frames"):
            value = require_integer_valued(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, init=False)
class EdgeQCConfig:
    """Composed configuration for VesEdge quality-control checks.

    Successful construction guarantees finite numeric QC thresholds within
    their allowed ranges and boolean enable/disable flags. QC execution code
    may therefore treat this object as validated configuration.

    New code supplies :class:`CurvatureQCConfig`, :class:`AreaQCConfig`, and
    :class:`InternalVesicleQCConfig` instances. The constructor and
    :meth:`from_dict` translate the former flat schema for compatibility.
    """

    curvature: CurvatureQCConfig
    area: AreaQCConfig = field(default_factory=AreaQCConfig)
    internal_vesicle: InternalVesicleQCConfig = field(
        default_factory=InternalVesicleQCConfig
    )

    def __init__(
        self,
        curvature: CurvatureQCConfig | float | None = None,
        area: AreaQCConfig | None = None,
        internal_vesicle: InternalVesicleQCConfig | None = None,
        **legacy_values,
    ) -> None:
        """Create a composed config, translating legacy flat arguments."""
        if isinstance(curvature, CurvatureQCConfig):
            if legacy_values:
                raise TypeError(
                    "Nested and legacy flat QC configuration cannot be mixed."
                )
            if area is not None and not isinstance(area, AreaQCConfig):
                raise TypeError("area must be an AreaQCConfig.")
            if internal_vesicle is not None and not isinstance(
                internal_vesicle, InternalVesicleQCConfig
            ):
                raise TypeError(
                    "internal_vesicle must be an InternalVesicleQCConfig."
                )
            object.__setattr__(self, "curvature", curvature)
            object.__setattr__(
                self, "area", area if area is not None else AreaQCConfig()
            )
            object.__setattr__(
                self,
                "internal_vesicle",
                internal_vesicle
                if internal_vesicle is not None
                else InternalVesicleQCConfig(),
            )
            return

        if area is not None or internal_vesicle is not None:
            raise TypeError(
                "Nested and legacy flat QC configuration cannot be mixed."
            )
        if curvature is not None:
            legacy_values["curvature_threshold"] = curvature
        migrated = self._from_legacy_dict(legacy_values)
        object.__setattr__(self, "curvature", migrated.curvature)
        object.__setattr__(self, "area", migrated.area)
        object.__setattr__(self, "internal_vesicle", migrated.internal_vesicle)

    @classmethod
    def from_dict(cls, values: dict) -> "EdgeQCConfig":
        """Deserialize nested configuration or migrate legacy flat values."""
        if "curvature" in values:
            return cls(
                curvature=CurvatureQCConfig(**values["curvature"]),
                area=AreaQCConfig(**values.get("area", {})),
                internal_vesicle=InternalVesicleQCConfig(
                    **values.get("internal_vesicle", {})
                ),
            )
        return cls._from_legacy_dict(values)

    @classmethod
    def _from_legacy_dict(cls, values: dict) -> "EdgeQCConfig":
        """Translate the former flat schema into composed configuration."""
        unknown = set(values) - {
            "curvature_threshold",
            "enable_curvature_qc",
            "max_relative_area_deviation",
            "enable_area_qc",
            "enable_internal_vesicle_qc",
            "max_internal_vesicle_area_fraction",
            "internal_vesicle_min_radius_ratio",
            "internal_vesicle_min_separation_pixels",
            "internal_vesicle_gradient_ratio",
            "internal_vesicle_max_radial_deviation_fraction",
            "internal_vesicle_min_angular_coverage",
            "internal_vesicle_max_frames",
            "internal_vesicle_min_valid_frames",
            "internal_vesicle_min_valid_frame_fraction",
            "internal_vesicle_min_frame_fraction",
        }
        if unknown:
            name = sorted(unknown)[0]
            raise TypeError(f"Unexpected QC configuration field: {name}")
        if "curvature_threshold" not in values:
            raise TypeError("curvature configuration is required.")
        return cls(
            curvature=CurvatureQCConfig(
                threshold=values["curvature_threshold"],
                enabled=values.get("enable_curvature_qc", True),
            ),
            area=AreaQCConfig(
                max_relative_deviation=values.get(
                    "max_relative_area_deviation", 0.25
                ),
                enabled=values.get("enable_area_qc", True),
            ),
            internal_vesicle=InternalVesicleQCConfig(
                enabled=values.get("enable_internal_vesicle_qc", False),
                max_area_fraction=values.get(
                    "max_internal_vesicle_area_fraction", 0.5
                ),
                min_radius_ratio=values.get(
                    "internal_vesicle_min_radius_ratio", 1.15
                ),
                min_separation_pixels=values.get(
                    "internal_vesicle_min_separation_pixels", 5.0
                ),
                gradient_ratio=values.get("internal_vesicle_gradient_ratio", 0.5),
                max_radial_deviation_fraction=values.get(
                    "internal_vesicle_max_radial_deviation_fraction", 0.15
                ),
                min_angular_coverage=values.get(
                    "internal_vesicle_min_angular_coverage", 0.6
                ),
                max_frames=values.get("internal_vesicle_max_frames", 20),
                min_valid_frames=values.get(
                    "internal_vesicle_min_valid_frames", 3
                ),
                min_valid_frame_fraction=values.get(
                    "internal_vesicle_min_valid_frame_fraction", 0.5
                ),
                min_frame_fraction=values.get(
                    "internal_vesicle_min_frame_fraction", 0.5
                ),
            ),
        )
