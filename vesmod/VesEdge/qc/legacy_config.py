"""Translate historical flat QC configurations into check-owned settings."""

from ..experimental.internal_vesicle_qc import InternalVesicleQCConfig
from .checks.area_qc import AreaQCConfig
from .checks.edge_filtering import CurvatureQCConfig


def migrate_flat_config(values: dict) -> dict:
    """Convert legacy QC settings while preserving old defaults."""
    legacy_fields = {
        "curvature_threshold", "enable_curvature_qc",
        "max_relative_area_deviation", "enable_area_qc",
        "enable_internal_vesicle_qc", "max_internal_vesicle_area_fraction",
        "internal_vesicle_min_radius_ratio",
        "internal_vesicle_min_separation_pixels",
        "internal_vesicle_min_separation_fraction",
        "internal_vesicle_gradient_ratio",
        "internal_vesicle_max_radial_deviation_fraction",
        "internal_vesicle_min_angular_coverage", "internal_vesicle_max_frames",
        "internal_vesicle_min_valid_frames",
        "internal_vesicle_min_valid_frame_fraction",
        "internal_vesicle_min_frame_fraction",
    }
    unknown = set(values) - legacy_fields
    if unknown:
        raise TypeError(f"Unexpected QC configuration field: {sorted(unknown)[0]}")
    if "internal_vesicle_min_separation_pixels" in values:
        raise ValueError(
            "Legacy internal_vesicle_min_separation_pixels cannot be converted "
            "without a contour radius; use internal_vesicle_min_separation_fraction."
        )
    values = {
        "curvature": CurvatureQCConfig(
            threshold=values["curvature_threshold"],
            enabled=values.get("enable_curvature_qc", True),
        ),
        "area": AreaQCConfig(
            max_relative_deviation=values.get("max_relative_area_deviation", 0.25),
            enabled=values.get("enable_area_qc", True),
        ),
        "internal_vesicle": InternalVesicleQCConfig(
            enabled=values.get("enable_internal_vesicle_qc", False),
            max_area_fraction=values.get("max_internal_vesicle_area_fraction", 0.5),
            min_radius_ratio=values.get("internal_vesicle_min_radius_ratio", 1.15),
            min_separation_fraction=values.get("internal_vesicle_min_separation_fraction", 0.4),
            gradient_ratio=values.get("internal_vesicle_gradient_ratio", 0.5),
            max_radial_deviation_fraction=values.get("internal_vesicle_max_radial_deviation_fraction", 0.15),
            min_angular_coverage=values.get("internal_vesicle_min_angular_coverage", 0.6),
            max_frames=values.get("internal_vesicle_max_frames", 20),
            min_valid_frames=values.get("internal_vesicle_min_valid_frames", 3),
            min_valid_frame_fraction=values.get("internal_vesicle_min_valid_frame_fraction", 0.5),
            min_frame_fraction=values.get("internal_vesicle_min_frame_fraction", 0.5),
        ),
    }
    return values
