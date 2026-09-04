"""Experimental measurement of resolvable structures inside vesicles.

This module is deliberately independent of edge extraction and quality
control.  Its thresholds are experimental and should be calibrated against
manually labelled videos before they are used to classify populations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.draw import polygon
from skimage.filters import sato
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk,
    skeletonize,
)

from vesmod.validation import (
    require_fraction,
    require_integer,
    require_positive_real,
)

from .models import ImageContour


@dataclass(frozen=True)
class InternalStructureConfig:
    """Parameters for experimental internal-structure detection.

    Parameters
    ----------
    membrane_exclusion_px : int
        Distance eroded from the detected contour, in pixels. This prevents
        the membrane and its optical blur from being counted as structure.
    background_sigma_px : float
        Gaussian sigma, in pixels, used to estimate smoothly varying interior
        intensity. It should be larger than the structures of interest.
    threshold_sigma : float
        Positive-residual threshold for high-confidence light-region seeds.
    min_region_area_px : int
        Minimum connected-component area retained as structure, in pixels.
    light_grow_sigma : float
        Positive-residual threshold through which bright seeds grow.
    min_light_circularity : float
        Minimum circularity retained as a bright internal vesicle.
    min_light_solidity : float
        Minimum filled fraction of a bright candidate's convex hull.
    max_light_eccentricity : float
        Maximum eccentricity retained as a circular or oval bright vesicle.
    structure_boundary_exclusion_px : int
        Distance from the detected outer contour excluded from structure
        candidates. This may be larger than ``membrane_exclusion_px`` so the
        membrane's optical profile cannot seed filaments or bubbles.
    filament_seed_threshold : float
        Sato vesselness required to seed a curvilinear structure.
    filament_grow_threshold : float
        Lower Sato vesselness through which curvilinear seeds may grow.
    filament_scales_px : tuple[float, ...]
        Gaussian scales used to calculate dark and light vesselness.
    min_filament_length_px : int
        Minimum connected skeleton length retained as a filament.
    bubble_edge_sigma : float
        Negative-residual magnitude required to seed a bubble boundary or
        compact dark-region mask. This can affect dark-region measurements
        and the merged structure output.
    bubble_edge_grow_sigma : float
        Lower negative-residual magnitude through which bubble boundaries and
        compact dark-region masks grow. Its absolute magnitude also gates
        curvilinear ridge evidence, so it can affect dark-region measurements
        and the merged structure output.
    bubble_closing_px : int
        Radius used to close small gaps in candidate bubble boundaries.
    min_bubble_area_px : int
        Minimum area retained for an enclosed bubble or compact dark-region
        mask. This can affect dark-region measurements and the merged
        structure output.
    min_bubble_boundary_fraction : float
        Minimum fraction of an enclosed boundary supported by dark pixels.
    min_bubble_circularity : float
        Minimum circularity retained for a dark-edged bubble or compact
        dark-region mask. This can affect dark-region measurements and the
        merged structure output.
    min_bubble_solidity : float
        Minimum filled fraction of a bubble or compact dark-region candidate's
        convex hull. This can affect dark-region measurements and the merged
        structure output.
    max_bubble_eccentricity : float
        Maximum eccentricity retained for a circular or oval bubble or compact
        dark-region mask. This can affect dark-region measurements and the
        merged structure output.
    max_bubble_area_fraction : float
        Maximum usable-interior fraction enclosed by one bubble.
    """

    membrane_exclusion_px: int = 5
    background_sigma_px: float = 30.0
    threshold_sigma: float = 4.0
    min_region_area_px: int = 9
    light_grow_sigma: float = 1.5
    min_light_circularity: float = 0.2
    min_light_solidity: float = 0.8
    max_light_eccentricity: float = 0.95
    structure_boundary_exclusion_px: int = 20
    filament_seed_threshold: float = 0.7
    filament_grow_threshold: float = 0.35
    filament_scales_px: tuple[float, ...] = (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    )
    min_filament_length_px: int = 20
    bubble_edge_sigma: float = 2.0
    bubble_edge_grow_sigma: float = 1.0
    bubble_closing_px: int = 4
    min_bubble_area_px: int = 100
    min_bubble_boundary_fraction: float = 0.45
    min_bubble_circularity: float = 0.2
    min_bubble_solidity: float = 0.8
    max_bubble_eccentricity: float = 0.95
    max_bubble_area_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration values and normalize scalar state."""
        integer_fields = (
            ("membrane_exclusion_px", False),
            ("min_region_area_px", True),
            ("structure_boundary_exclusion_px", False),
            ("min_filament_length_px", True),
            ("bubble_closing_px", False),
            ("min_bubble_area_px", True),
        )
        for name, must_be_positive in integer_fields:
            value = require_integer(getattr(self, name), name)
            if must_be_positive and value <= 0:
                raise ValueError(f"{name} must be positive.")
            if not must_be_positive and value < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)

        positive_real_fields = (
            "background_sigma_px",
            "threshold_sigma",
            "light_grow_sigma",
            "filament_seed_threshold",
            "filament_grow_threshold",
            "bubble_edge_sigma",
            "bubble_edge_grow_sigma",
        )
        for name in positive_real_fields:
            value = require_positive_real(
                getattr(self, name),
                name,
                finite_message=f"{name} must be finite and positive.",
                range_message=f"{name} must be finite and positive.",
            )
            object.__setattr__(self, name, value)

        fraction_fields = (
            "min_light_circularity",
            "min_light_solidity",
            "max_light_eccentricity",
            "min_bubble_boundary_fraction",
            "min_bubble_circularity",
            "min_bubble_solidity",
            "max_bubble_eccentricity",
            "max_bubble_area_fraction",
        )
        for name in fraction_fields:
            value = require_fraction(
                getattr(self, name),
                name,
                range_message=f"{name} must be between zero and one.",
            )
            object.__setattr__(self, name, value)

        if self.light_grow_sigma > self.threshold_sigma:
            raise ValueError("light_grow_sigma cannot exceed threshold_sigma.")
        if self.structure_boundary_exclusion_px < self.membrane_exclusion_px:
            raise ValueError(
                "structure_boundary_exclusion_px cannot be smaller than "
                "membrane_exclusion_px."
            )
        if self.filament_grow_threshold > self.filament_seed_threshold:
            raise ValueError(
                "filament_grow_threshold cannot exceed "
                "filament_seed_threshold."
            )
        if not isinstance(self.filament_scales_px, tuple):
            raise TypeError("filament_scales_px must be a tuple.")
        if not self.filament_scales_px:
            raise ValueError("filament_scales_px cannot be empty.")
        filament_scales = tuple(
            require_positive_real(
                scale,
                "filament_scales_px",
                finite_message="filament_scales_px must be finite and positive.",
                range_message="filament_scales_px must be finite and positive.",
            )
            for scale in self.filament_scales_px
        )
        object.__setattr__(self, "filament_scales_px", filament_scales)
        if self.bubble_edge_grow_sigma > self.bubble_edge_sigma:
            raise ValueError(
                "bubble_edge_grow_sigma cannot exceed bubble_edge_sigma."
            )
        if self.max_bubble_area_fraction == 0.0:
            raise ValueError("max_bubble_area_fraction must be positive.")


@dataclass(frozen=True)
class InternalStructureRegion:
    """One merged structure region in original-image coordinates.

    ``evidence_types`` records which proposal generators support the region;
    it is diagnostic provenance rather than a biological classification.
    """

    label: int
    area_px: int
    centroid_yx: tuple[float, float]
    bbox_yx: tuple[int, int, int, int]
    mean_signed_residual: float
    structure_type: str = "unclassified"
    skeleton_length_px: int = 0
    evidence_types: tuple[str, ...] = ()

    @property
    def polarity(self) -> str:
        """Return the mean intensity polarity of this merged region."""
        return "bright" if self.mean_signed_residual >= 0 else "dark"


@dataclass(frozen=True)
class InternalStructureFrameResult:
    """Detection products for one frame, stored in a bounded image crop.

    ``crop_origin_yx`` locates the crop in the source image. Use
    :meth:`to_full_frame_mask` when an image-aligned mask is needed.
    """

    original_shape: tuple[int, int]
    crop_origin_yx: tuple[int, int]
    usable_interior_mask: NDArray[np.bool_]
    residual: NDArray[np.float64]
    structure_mask: NDArray[np.bool_]
    regions: tuple[InternalStructureRegion, ...]
    noise_sigma: float
    light_region_mask: NDArray[np.bool_] | None = None
    dark_filament_mask: NDArray[np.bool_] | None = None
    bubble_region_mask: NDArray[np.bool_] | None = None
    dark_filament_skeleton: NDArray[np.bool_] | None = None
    dark_region_mask: NDArray[np.bool_] | None = None

    @property
    def usable_area_px(self) -> int:
        """Return the number of interior pixels included in analysis."""
        return int(np.count_nonzero(self.usable_interior_mask))

    @property
    def structured_area_px(self) -> int:
        """Return the number of interior pixels classified as structure."""
        return int(np.count_nonzero(self.structure_mask))

    @property
    def structured_area_fraction(self) -> float:
        """Return detected structure area divided by usable interior area."""
        if self.usable_area_px == 0:
            return 0.0
        return self.structured_area_px / self.usable_area_px

    def _channel_mask(
        self,
        mask: NDArray[np.bool_] | None,
    ) -> NDArray[np.bool_]:
        """Return an empty compatible mask for legacy results without channels."""
        if mask is None:
            return np.zeros_like(self.structure_mask)
        return mask

    @property
    def light_area_fraction(self) -> float:
        """Return the usable-interior fraction assigned to light regions."""
        return self._area_fraction(self._channel_mask(self.light_region_mask))

    @property
    def dark_region_area_fraction(self) -> float:
        """Return the fraction supported by compact dark-region evidence."""
        return self._area_fraction(self._channel_mask(self.dark_region_mask))

    @property
    def filament_area_fraction(self) -> float:
        """Return the usable-interior fraction assigned to dark filaments."""
        return self._area_fraction(self._channel_mask(self.dark_filament_mask))

    @property
    def bubble_area_fraction(self) -> float:
        """Return the usable-interior fraction enclosed by detected bubbles."""
        return self._area_fraction(self._channel_mask(self.bubble_region_mask))

    @property
    def filament_length_px(self) -> int:
        """Return total skeleton length as a pixel-count approximation."""
        return int(
            np.count_nonzero(
                self._channel_mask(self.dark_filament_skeleton)
            )
        )

    @property
    def bubble_count(self) -> int:
        """Return the number of enclosed-boundary evidence regions."""
        return int(label(self._channel_mask(self.bubble_region_mask)).max())

    @property
    def structure_count(self) -> int:
        """Return the number of merged connected structure regions."""
        return len(self.regions)

    def _area_fraction(self, mask: NDArray[np.bool_]) -> float:
        """Return mask area divided by usable interior area."""
        if self.usable_area_px == 0:
            return 0.0
        return float(np.count_nonzero(mask)) / self.usable_area_px

    def to_full_frame_mask(self) -> NDArray[np.bool_]:
        """Map the cropped structure mask back to the original image."""
        return self._to_full_frame_mask(self.structure_mask)

    def to_full_frame_channel_mask(
        self,
        structure_type: str,
    ) -> NDArray[np.bool_]:
        """Map one named structure channel back to the original image.

        Parameters
        ----------
        structure_type : str
            One of light_region, dark_region, dark_filament, or bubble.
        """
        channel_masks = {
            "light_region": self.light_region_mask,
            "dark_region": self.dark_region_mask,
            "dark_filament": self.dark_filament_mask,
            "bubble": self.bubble_region_mask,
        }
        if structure_type not in channel_masks:
            expected = ", ".join(channel_masks)
            raise ValueError(
                f"Unknown structure_type {structure_type!r}; expected {expected}."
            )
        mask = self._channel_mask(channel_masks[structure_type])
        return self._to_full_frame_mask(mask)

    def _to_full_frame_mask(
        self,
        crop_mask: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        """Map one cropped mask back to the original image."""
        full_mask = np.zeros(self.original_shape, dtype=bool)
        y_start, x_start = self.crop_origin_yx
        y_stop = y_start + crop_mask.shape[0]
        x_stop = x_start + crop_mask.shape[1]
        full_mask[y_start:y_stop, x_start:x_stop] = crop_mask
        return full_mask


@dataclass(frozen=True)
class InternalStructureVideoSummary:
    """Framewise internal-structure abundance summarized over one video."""

    median_area_fraction: float
    upper_area_fraction: float
    frame_prevalence: float
    n_frames: int
    median_light_area_fraction: float = 0.0
    median_dark_region_area_fraction: float = 0.0
    median_filament_area_fraction: float = 0.0
    median_filament_length_px: float = 0.0
    median_bubble_area_fraction: float = 0.0
    median_bubble_count: float = 0.0


def detect_internal_structures(
    frame: NDArray[np.generic],
    contour: ImageContour,
    config: InternalStructureConfig | None = None,
) -> InternalStructureFrameResult:
    """Detect a merged mask of resolvable internal structure evidence.

    Detection is performed in a crop around ``contour``. The returned masks
    remain in crop coordinates for compactness, while regions and
    :meth:`InternalStructureFrameResult.to_full_frame_mask` use the original
    image coordinate system.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 2:
        raise ValueError("frame must be a two-dimensional numpy array.")
    if not np.issubdtype(frame.dtype, np.number):
        raise TypeError("frame must contain numeric intensities.")
    if not np.all(np.isfinite(frame)):
        raise ValueError("frame intensities must be finite.")

    settings = config or InternalStructureConfig()
    image = frame.astype(np.float64, copy=False)
    crop, interior_mask, crop_origin = _crop_to_contour(image, contour)
    usable_mask = _exclude_membrane(interior_mask, settings)
    if not np.any(usable_mask):
        raise ValueError(
            "Membrane exclusion leaves no usable vesicle interior."
        )
    detection_mask = _exclude_structure_boundary(interior_mask, settings)
    if not np.any(detection_mask):
        raise ValueError(
            "Structure-boundary exclusion leaves no detection interior."
        )

    background = _masked_gaussian_background(
        crop,
        usable_mask,
        settings.background_sigma_px,
    )
    residual = np.zeros_like(crop, dtype=np.float64)
    residual[usable_mask] = crop[usable_mask] - background[usable_mask]
    noise_sigma = _robust_sigma(residual[usable_mask])

    if noise_sigma == 0.0:
        light_mask = np.zeros_like(usable_mask)
        dark_mask = np.zeros_like(usable_mask)
        ridge_mask = np.zeros_like(usable_mask)
        ridge_skeleton = np.zeros_like(usable_mask)
        bubble_mask = np.zeros_like(usable_mask)
    else:
        normalized = np.zeros_like(residual)
        normalized[usable_mask] = residual[usable_mask] / noise_sigma
        light_mask = _detect_light_regions(
            normalized,
            detection_mask,
            usable_mask,
            settings,
        )
        dark_mask = _detect_dark_regions(
            normalized,
            detection_mask,
            usable_mask,
            settings,
        )
        ridge_input = np.where(detection_mask, normalized, 0.0)
        dark_ridge_response = _ridge_response(
            ridge_input,
            settings.filament_scales_px,
            black_ridges=True,
        )
        light_ridge_response = _ridge_response(
            ridge_input,
            settings.filament_scales_px,
            black_ridges=False,
        )
        dark_ridge_neighborhood = ndimage.binary_dilation(
            detection_mask
            & (normalized <= -settings.bubble_edge_grow_sigma),
            structure=disk(int(np.ceil(max(settings.filament_scales_px)))),
        )
        paired_light_response = np.where(
            dark_ridge_neighborhood,
            light_ridge_response,
            0.0,
        )
        ridge_response = np.maximum(
            dark_ridge_response,
            paired_light_response,
        )
        ridge_response[np.abs(normalized) < settings.bubble_edge_grow_sigma] = 0.0
        ridge_mask, ridge_skeleton = _detect_curvilinear_structures(
            ridge_response,
            detection_mask,
            detection_mask,
            settings,
        )
        bubble_mask = _detect_bubbles(
            normalized,
            dark_ridge_response,
            ridge_mask,
            detection_mask,
            usable_mask,
            settings,
        )
        (
            dark_mask,
            ridge_mask,
            ridge_skeleton,
            bubble_mask,
        ) = _suppress_bright_region_halos(
            light_mask,
            dark_mask,
            ridge_mask,
            bubble_mask,
            settings,
        )
        (
            light_mask,
            dark_mask,
            ridge_mask,
            ridge_skeleton,
        ) = _suppress_enclosed_boundary_halos(
            bubble_mask,
            light_mask,
            dark_mask,
            ridge_mask,
            settings,
        )

    evidence_masks = {
        "bright_region": light_mask,
        "dark_region": dark_mask,
        "curvilinear": ridge_mask,
        "enclosed_boundary": bubble_mask,
    }
    structure_mask = np.logical_or.reduce(tuple(evidence_masks.values()))
    regions = _describe_merged_regions(
        label(structure_mask, connectivity=2),
        residual,
        crop_origin,
        evidence_masks,
        ridge_skeleton,
    )
    return InternalStructureFrameResult(
        original_shape=frame.shape,
        crop_origin_yx=crop_origin,
        usable_interior_mask=usable_mask,
        residual=residual,
        structure_mask=structure_mask,
        regions=regions,
        noise_sigma=noise_sigma,
        light_region_mask=light_mask,
        dark_region_mask=dark_mask,
        dark_filament_mask=ridge_mask,
        bubble_region_mask=bubble_mask,
        dark_filament_skeleton=ridge_skeleton,
    )


def summarize_internal_structures(
    results: list[InternalStructureFrameResult]
    | tuple[InternalStructureFrameResult, ...],
) -> InternalStructureVideoSummary:
    """Summarize framewise abundance without tracking moving structures."""
    if not results:
        raise ValueError("At least one frame result is required.")
    fractions = np.asarray(
        [result.structured_area_fraction for result in results],
        dtype=np.float64,
    )
    return InternalStructureVideoSummary(
        median_area_fraction=float(np.median(fractions)),
        upper_area_fraction=float(np.quantile(fractions, 0.9)),
        frame_prevalence=float(np.mean(fractions > 0.0)),
        n_frames=len(results),
        median_light_area_fraction=float(
            np.median([result.light_area_fraction for result in results])
        ),
        median_dark_region_area_fraction=float(
            np.median(
                [result.dark_region_area_fraction for result in results]
            )
        ),
        median_filament_area_fraction=float(
            np.median([result.filament_area_fraction for result in results])
        ),
        median_filament_length_px=float(
            np.median([result.filament_length_px for result in results])
        ),
        median_bubble_area_fraction=float(
            np.median([result.bubble_area_fraction for result in results])
        ),
        median_bubble_count=float(
            np.median([result.bubble_count for result in results])
        ),
    )


def _crop_to_contour(
    image: NDArray[np.float64],
    contour: ImageContour,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], tuple[int, int]]:
    """Return a clipped contour crop and its rasterized interior mask."""
    x_vals = np.asarray(contour.x[:-1], dtype=np.float64)
    y_vals = np.asarray(contour.y[:-1], dtype=np.float64)
    if (
        x_vals.size < 3
        or y_vals.size != x_vals.size
        or not np.all(np.isfinite(x_vals))
        or not np.all(np.isfinite(y_vals))
    ):
        raise ValueError("contour must contain at least three finite points.")

    height, width = image.shape
    y_start = max(0, int(np.floor(np.min(y_vals))))
    y_stop = min(height, int(np.ceil(np.max(y_vals))) + 1)
    x_start = max(0, int(np.floor(np.min(x_vals))))
    x_stop = min(width, int(np.ceil(np.max(x_vals))) + 1)
    if y_start >= y_stop or x_start >= x_stop:
        raise ValueError("contour does not overlap the image.")

    crop = image[y_start:y_stop, x_start:x_stop]
    interior_mask = np.zeros(crop.shape, dtype=bool)
    rows, columns = polygon(
        y_vals - y_start,
        x_vals - x_start,
        shape=crop.shape,
    )
    interior_mask[rows, columns] = True
    return crop, interior_mask, (y_start, x_start)


def _exclude_membrane(
    interior_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> NDArray[np.bool_]:
    """Erode the contour mask by the configured membrane margin."""
    if config.membrane_exclusion_px == 0:
        return interior_mask.copy()
    return ndimage.binary_erosion(
        interior_mask,
        structure=disk(config.membrane_exclusion_px),
    )


def _exclude_structure_boundary(
    interior_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> NDArray[np.bool_]:
    """Return the interior eligible to contain structure candidates.

    Multiscale ridge filters respond over several Gaussian widths.  A fixed
    margin smaller than that support can therefore admit the inward optical
    shadow of the outer membrane even though the membrane itself is masked.
    Expand the requested margin to cover four times the largest ridge scale,
    while limiting the automatic expansion to one half of the vesicle's
    inradius so small vesicles retain a useful detection interior.
    """
    inradius = float(np.max(ndimage.distance_transform_edt(interior_mask)))
    scale_margin = int(np.ceil(4.0 * max(config.filament_scales_px)))
    size_limited_margin = min(scale_margin, int(np.floor(0.5 * inradius)))
    exclusion_px = max(
        config.structure_boundary_exclusion_px,
        size_limited_margin,
    )
    if exclusion_px == 0:
        return interior_mask.copy()
    return ndimage.binary_erosion(
        interior_mask,
        structure=disk(exclusion_px),
    )


def _masked_gaussian_background(
    image: NDArray[np.float64],
    mask: NDArray[np.bool_],
    sigma: float,
) -> NDArray[np.float64]:
    """Estimate smooth intensity without mixing exterior pixels into it."""
    weights = ndimage.gaussian_filter(mask.astype(np.float64), sigma=sigma)
    weighted_image = ndimage.gaussian_filter(
        image * mask,
        sigma=sigma,
    )
    background = np.zeros_like(image, dtype=np.float64)
    np.divide(
        weighted_image,
        weights,
        out=background,
        where=weights > np.finfo(np.float64).eps,
    )
    return background


def _robust_sigma(values: NDArray[np.float64]) -> float:
    """Estimate residual noise using the scaled median absolute deviation."""
    median = np.median(values)
    sigma = 1.4826 * np.median(np.abs(values - median))
    if sigma == 0.0:
        sigma = float(np.std(values))
    return float(sigma)


def _detect_light_regions(
    normalized_residual: NDArray[np.float64],
    seed_mask: NDArray[np.bool_],
    growth_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> NDArray[np.bool_]:
    """Return compact positive-residual structure proposals."""
    return _detect_compact_regions(
        normalized_residual,
        seed_mask,
        growth_mask,
        seed_sigma=config.threshold_sigma,
        grow_sigma=config.light_grow_sigma,
        polarity=1,
        min_area_px=config.min_region_area_px,
        min_circularity=config.min_light_circularity,
        min_solidity=config.min_light_solidity,
        max_eccentricity=config.max_light_eccentricity,
    )


def _detect_dark_regions(
    normalized_residual: NDArray[np.float64],
    seed_mask: NDArray[np.bool_],
    growth_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> NDArray[np.bool_]:
    """Return compact negative-residual structure proposals.

    This proposal catches filled dark objects that do not contain the neutral
    hole required by the enclosed-boundary detector.
    """
    return _detect_compact_regions(
        normalized_residual,
        seed_mask,
        growth_mask,
        seed_sigma=config.bubble_edge_sigma,
        grow_sigma=config.bubble_edge_grow_sigma,
        polarity=-1,
        min_area_px=config.min_bubble_area_px,
        min_circularity=config.min_bubble_circularity,
        min_solidity=config.min_bubble_solidity,
        max_eccentricity=config.max_bubble_eccentricity,
    )


def _detect_compact_regions(
    normalized_residual: NDArray[np.float64],
    seed_mask: NDArray[np.bool_],
    growth_mask: NDArray[np.bool_],
    *,
    seed_sigma: float,
    grow_sigma: float,
    polarity: int,
    min_area_px: int,
    min_circularity: float,
    min_solidity: float,
    max_eccentricity: float,
) -> NDArray[np.bool_]:
    """Grow and shape-filter compact signed-residual proposals."""
    signed_residual = polarity * normalized_residual
    seeds = seed_mask & (signed_residual >= seed_sigma)
    candidates = growth_mask & (signed_residual >= grow_sigma)
    grown = ndimage.binary_propagation(seeds, mask=candidates)
    return _retain_compact_components(
        grown,
        min_area_px,
        min_circularity,
        min_solidity,
        max_eccentricity,
    )


def _retain_compact_components(
    mask: NDArray[np.bool_],
    min_area_px: int,
    min_circularity: float,
    min_solidity: float,
    max_eccentricity: float,
) -> NDArray[np.bool_]:
    """Reapply compact-component size and shape requirements to a mask."""
    filtered = _remove_small_components(mask, min_size=min_area_px)
    accepted = np.zeros_like(filtered)
    for component in regionprops(label(filtered, connectivity=2)):
        if not _component_shape_passes(
            component,
            min_circularity,
            min_solidity,
            max_eccentricity,
        ):
            continue
        coordinates = component.coords
        accepted[coordinates[:, 0], coordinates[:, 1]] = True
    return accepted


def _suppress_bright_region_halos(
    bright_mask: NDArray[np.bool_],
    dark_mask: NDArray[np.bool_],
    ridge_mask: NDArray[np.bool_],
    enclosed_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> tuple[
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    """Remove secondary evidence caused by a compact bright structure.

    Gaussian background subtraction produces a negative halo around a strong
    positive object.  Without this suppression, the halo can be proposed a
    second time as a dark ridge or closed boundary and artificially enlarge
    the authoritative union.  Compact bright evidence remains untouched.

    Light borders alongside elongated dark structures are not affected: they
    fail the compact-region shape checks and therefore do not enter
    ``bright_mask``.
    """
    if not np.any(bright_mask):
        return dark_mask, ridge_mask, skeletonize(ridge_mask), enclosed_mask

    halo_radius = int(np.ceil(4.0 * max(config.filament_scales_px)))
    bright_neighborhood = ndimage.binary_dilation(
        bright_mask,
        structure=disk(halo_radius),
    )
    dark_without_halo = _retain_compact_components(
        dark_mask & ~bright_neighborhood,
        config.min_bubble_area_px,
        config.min_bubble_circularity,
        config.min_bubble_solidity,
        config.max_bubble_eccentricity,
    )
    ridge_without_halo = ridge_mask & ~bright_neighborhood
    ridge_without_halo, ridge_skeleton = _retain_curvilinear_components(
        ridge_without_halo,
        config.min_filament_length_px,
        int(np.ceil(max(config.filament_scales_px))),
    )

    enclosed_without_bright = np.zeros_like(enclosed_mask)
    for component in regionprops(label(enclosed_mask, connectivity=2)):
        coordinates = component.coords
        if np.any(bright_mask[coordinates[:, 0], coordinates[:, 1]]):
            continue
        enclosed_without_bright[coordinates[:, 0], coordinates[:, 1]] = True

    return (
        dark_without_halo,
        ridge_without_halo,
        ridge_skeleton,
        enclosed_without_bright,
    )


def _suppress_enclosed_boundary_halos(
    enclosed_mask: NDArray[np.bool_],
    bright_mask: NDArray[np.bool_],
    dark_mask: NDArray[np.bool_],
    ridge_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> tuple[
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.bool_],
]:
    """Remove redundant structure evidence immediately around bubbles.

    A resolved bubble is represented by its filled enclosed-boundary mask.
    Optical ringing and the multiscale filters can otherwise add a second
    skirt of bright, dark, or curvilinear evidence around the same object.
    Suppress that skirt over one maximum ridge scale while leaving the
    enclosed-boundary evidence itself unchanged. Compact channels are
    revalidated after masking so halo subtraction cannot leave fragments that
    no longer satisfy their configured size or shape requirements.
    """
    if not np.any(enclosed_mask):
        return bright_mask, dark_mask, ridge_mask, skeletonize(ridge_mask)

    halo_radius = int(np.ceil(max(config.filament_scales_px))) + 1
    bubble_neighborhood = ndimage.binary_dilation(
        enclosed_mask,
        structure=disk(halo_radius),
    )
    bright_without_halo = _retain_compact_components(
        bright_mask & ~bubble_neighborhood,
        config.min_region_area_px,
        config.min_light_circularity,
        config.min_light_solidity,
        config.max_light_eccentricity,
    )
    dark_without_halo = _retain_compact_components(
        dark_mask & ~bubble_neighborhood,
        config.min_bubble_area_px,
        config.min_bubble_circularity,
        config.min_bubble_solidity,
        config.max_bubble_eccentricity,
    )
    ridge_without_halo = ridge_mask & ~bubble_neighborhood
    ridge_without_halo, ridge_skeleton = _retain_curvilinear_components(
        ridge_without_halo,
        config.min_filament_length_px,
        halo_radius,
    )
    return (
        bright_without_halo,
        dark_without_halo,
        ridge_without_halo,
        ridge_skeleton,
    )


def _retain_curvilinear_components(
    mask: NDArray[np.bool_],
    minimum_length_px: int,
    maximum_radius_px: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Reapply the skeleton-length requirement after masking a ridge."""
    skeleton = skeletonize(mask)
    kept_skeleton = np.zeros_like(skeleton)
    for component in regionprops(label(skeleton, connectivity=2)):
        if component.area < minimum_length_px:
            continue
        coordinates = component.coords
        kept_skeleton[coordinates[:, 0], coordinates[:, 1]] = True
    if not np.any(kept_skeleton):
        return np.zeros_like(mask), kept_skeleton
    kept_mask = mask & ndimage.binary_dilation(
        kept_skeleton,
        structure=disk(maximum_radius_px),
    )
    return kept_mask, skeletonize(kept_mask)


def _ridge_response(
    normalized_residual: NDArray[np.float64],
    scales: tuple[float, ...],
    *,
    black_ridges: bool,
) -> NDArray[np.float64]:
    """Return multiscale vesselness for one intensity polarity."""
    return sato(
        normalized_residual,
        sigmas=scales,
        black_ridges=black_ridges,
    )


def _detect_curvilinear_structures(
    ridge_response: NDArray[np.float64],
    seed_mask: NDArray[np.bool_],
    growth_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Detect connected dark-or-light ridges with sufficient length."""
    seeds = seed_mask & (
        ridge_response >= config.filament_seed_threshold
    )
    candidates = growth_mask & (
        ridge_response >= config.filament_grow_threshold
    )
    candidates = ndimage.binary_propagation(seeds, mask=candidates)
    return _retain_curvilinear_components(
        candidates,
        config.min_filament_length_px,
        int(np.ceil(max(config.filament_scales_px))),
    )


def _detect_bubbles(
    normalized_residual: NDArray[np.float64],
    ridge_response: NDArray[np.float64],
    retained_ridge_mask: NDArray[np.bool_],
    detection_mask: NDArray[np.bool_],
    usable_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> NDArray[np.bool_]:
    """Detect dark, sufficiently closed boundaries and fill their interiors.

    Bubble edges are ridges, just like filaments.  The distinction is
    topological: a bubble ridge encloses a compact interior.  Residual-based,
    dark-ridge, and retained curvilinear edges are evaluated separately so
    unrelated texture from different proposal generators cannot bridge two
    candidates into one artificial enclosure.  The retained mask recovers
    mixed-polarity optical rings whose dark boundary alone is interrupted.
    """
    dark_seeds = detection_mask & (
        normalized_residual <= -config.bubble_edge_sigma
    )
    dark_candidates = detection_mask & (
        normalized_residual <= -config.bubble_edge_grow_sigma
    )
    dark_edge = ndimage.binary_propagation(
        dark_seeds,
        mask=dark_candidates,
    )
    ridge_candidates = (
        detection_mask
        & (normalized_residual < 0.0)
        & (ridge_response >= config.filament_grow_threshold)
    )
    ridge_seed_threshold = np.mean(
        (config.filament_seed_threshold, config.filament_grow_threshold)
    )
    ridge_seeds = ridge_candidates & (
        ridge_response >= ridge_seed_threshold
    )
    ridge_edge = ndimage.binary_propagation(
        ridge_seeds,
        mask=ridge_candidates,
    )
    bubble_mask = _bubbles_enclosed_by_edge(
        dark_edge,
        detection_mask,
        usable_mask,
        config,
    )
    bubble_mask |= _bubbles_enclosed_by_edge(
        ridge_edge,
        detection_mask,
        usable_mask,
        config,
    )
    bubble_mask |= _bubbles_enclosed_by_edge(
        retained_ridge_mask,
        detection_mask,
        usable_mask,
        config,
    )
    return bubble_mask


def _bubbles_enclosed_by_edge(
    dark_edge: NDArray[np.bool_],
    detection_mask: NDArray[np.bool_],
    usable_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> NDArray[np.bool_]:
    """Return plausible bubble interiors enclosed by one edge-evidence map."""
    if config.bubble_closing_px:
        closed_edge = ndimage.binary_closing(
            dark_edge,
            structure=disk(config.bubble_closing_px),
        )
    else:
        closed_edge = dark_edge
    filled = ndimage.binary_fill_holes(closed_edge) & detection_mask
    enclosed = filled & ~closed_edge
    bubble_mask = np.zeros_like(usable_mask)
    usable_area = np.count_nonzero(usable_mask)
    for candidate in regionprops(label(enclosed, connectivity=2)):
        if candidate.area < config.min_bubble_area_px:
            continue
        if candidate.area / usable_area > config.max_bubble_area_fraction:
            continue
        if not _component_shape_passes(
            candidate,
            config.min_bubble_circularity,
            config.min_bubble_solidity,
            config.max_bubble_eccentricity,
        ):
            continue
        interior = np.zeros_like(usable_mask)
        coordinates = candidate.coords
        interior[coordinates[:, 0], coordinates[:, 1]] = True
        boundary = ndimage.binary_dilation(
            interior,
            structure=disk(1),
        ) & ~interior
        boundary &= detection_mask
        boundary_size = np.count_nonzero(boundary)
        if boundary_size == 0:
            continue
        coverage = np.count_nonzero(boundary & dark_edge) / boundary_size
        if coverage < config.min_bubble_boundary_fraction:
            continue
        bubble_mask |= interior | (boundary & closed_edge)
    return bubble_mask


def _component_shape_passes(
    component,
    min_circularity: float,
    min_solidity: float,
    max_eccentricity: float,
) -> bool:
    """Return whether a connected component is plausibly circular or oval."""
    perimeter = component.perimeter_crofton
    circularity = (
        0.0
        if perimeter == 0.0
        else 4.0 * np.pi * component.area / perimeter**2
    )
    return (
        circularity >= min_circularity
        and component.solidity >= min_solidity
        and component.eccentricity <= max_eccentricity
    )


def _remove_small_components(
    mask: NDArray[np.bool_],
    min_size: int,
) -> NDArray[np.bool_]:
    """Remove connected components containing fewer than min_size pixels."""
    labelled = label(mask, connectivity=2)
    kept = np.zeros_like(mask)
    for component in regionprops(labelled):
        if component.area >= min_size:
            coordinates = component.coords
            kept[coordinates[:, 0], coordinates[:, 1]] = True
    return kept


def _describe_merged_regions(
    labelled: NDArray[np.int_],
    residual: NDArray[np.float64],
    crop_origin: tuple[int, int],
    evidence_masks: dict[str, NDArray[np.bool_]],
    ridge_skeleton: NDArray[np.bool_],
) -> tuple[InternalStructureRegion, ...]:
    """Convert merged regions and their supporting evidence to measurements."""
    y_offset, x_offset = crop_origin
    descriptions = []
    for region in regionprops(labelled, intensity_image=residual):
        min_y, min_x, max_y, max_x = region.bbox
        centroid_y, centroid_x = region.centroid
        coordinates = region.coords
        rows = coordinates[:, 0]
        columns = coordinates[:, 1]
        evidence_types = tuple(
            name
            for name, mask in evidence_masks.items()
            if np.any(mask[rows, columns])
        )
        descriptions.append(
            InternalStructureRegion(
                label=region.label,
                area_px=int(region.area),
                centroid_yx=(
                    centroid_y + y_offset,
                    centroid_x + x_offset,
                ),
                bbox_yx=(
                    min_y + y_offset,
                    min_x + x_offset,
                    max_y + y_offset,
                    max_x + x_offset,
                ),
                mean_signed_residual=float(region.intensity_mean),
                structure_type="structure",
                skeleton_length_px=int(
                    np.count_nonzero(ridge_skeleton[rows, columns])
                ),
                evidence_types=evidence_types,
            )
        )
    return tuple(descriptions)
