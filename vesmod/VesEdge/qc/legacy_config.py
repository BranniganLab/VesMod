"""Translate historical flat QC configurations into check-owned settings."""

from ..experimental.area_qc import AreaQCConfig
from .checks.edge_filtering import CurvatureQCConfig
from .checks.localized_deviation_qc import LocalizedDeviationQCConfig


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
    if values.get("enable_internal_vesicle_qc", False):
        raise ValueError("Inner-vesicle selection QC was removed and cannot be replayed.")
    values = {
        "curvature": CurvatureQCConfig(
            threshold=values["curvature_threshold"],
            enabled=values.get("enable_curvature_qc", True),
        ),
        "area": AreaQCConfig(
            max_relative_deviation=values.get("max_relative_area_deviation", 0.25),
            enabled=values.get("enable_area_qc", True),
        ),
        "localized_deviation": LocalizedDeviationQCConfig(enabled=False),
    }
    return values


def discard_removed_check(values: dict) -> None:
    """Ignore recorded disabled settings for the deleted check."""
    if "internal_vesicle" not in values:
        return
    settings = values.pop("internal_vesicle")
    if not isinstance(settings, dict) or settings.get("enabled", False) is not False:
        raise ValueError("Inner-vesicle selection QC was removed and cannot be replayed.")
