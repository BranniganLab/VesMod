#!/usr/bin/env python3
"""Configuration façade for VesEdge extraction and quality control."""

from dataclasses import dataclass

from vesmod.validation import require_integer_valued, require_positive_real


_CALIBRATION_SOURCES = {"measured", "assumed", "unspecified"}


@dataclass(frozen=True)
class EdgeExtractionConfig:
    """Validated settings used while extracting and preparing contours."""

    pixels_per_micron: float = 1
    n_angular_samples: int | None = 120
    calibration_source: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pixels_per_micron",
            require_positive_real(self.pixels_per_micron, "pixels_per_micron"),
        )
        if self.calibration_source not in _CALIBRATION_SOURCES:
            allowed = ", ".join(sorted(_CALIBRATION_SOURCES))
            raise ValueError(f"calibration_source must be one of: {allowed}.")
        if self.n_angular_samples is None:
            return
        count = require_integer_valued(self.n_angular_samples, "n_angular_samples")
        if count <= 0:
            raise ValueError("n_angular_samples must be positive.")
        object.__setattr__(self, "n_angular_samples", count)


# These re-exports are a public convenience only. Concrete QC schemas live
# beside the checks that validate and consume them.
from .area_qc import AreaQCConfig  # noqa: E402
from .edge_filtering import CurvatureQCConfig  # noqa: E402
from .experimental.internal_vesicle_qc import InternalVesicleQCConfig  # noqa: E402
from .localized_deviation_qc import LocalizedDeviationQCConfig  # noqa: E402
from .minimum_radius_qc import MinimumRadiusQCConfig  # noqa: E402
from .qc_config import EdgeQCConfig  # noqa: E402
from .singleton_qc import SingletonDeviationQCConfig  # noqa: E402
