"""Experimental measurement of resolvable structures inside vesicles.

This module is deliberately independent of edge extraction and quality
control.  Its thresholds are experimental and should be calibrated against
manually labelled videos before they are used to classify populations.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.draw import polygon
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    disk,
    remove_small_objects,
    skeletonize,
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
    """

    membrane_exclusion_px: int = 5
    background_sigma_px: float = 30.0
    threshold_sigma: float = 4.0
    min_region_area_px: int = 9
    light_grow_sigma: float = 1.5
    filament_threshold_sigma: float = 1.5
    filament_scales_px: tuple[float, ...] = (1.0, 2.0, 3.0)
    min_filament_length_px: int = 8
    bubble_edge_sigma: float = 2.0
    bubble_closing_px: int = 2
    min_bubble_area_px: int = 25
    min_bubble_boundary_fraction: float = 0.45

    def __post_init__(self) -> None:
        """Validate configuration values."""
        _validate_nonnegative_integer(
            self.membrane_exclusion_px,
            "membrane_exclusion_px",
        )
        _validate_positive_real(
            self.background_sigma_px,
            "background_sigma_px",
        )
        _validate_positive_real(self.threshold_sigma, "threshold_sigma")
        _validate_positive_integer(
            self.min_region_area_px,
            "min_region_area_px",
        )
        _validate_positive_real(self.light_grow_sigma, "light_grow_sigma")
        if self.light_grow_sigma > self.threshold_sigma:
            raise ValueError("light_grow_sigma cannot exceed threshold_sigma.")
        _validate_positive_real(
            self.filament_threshold_sigma,
            "filament_threshold_sigma",
        )
        if not isinstance(self.filament_scales_px, tuple):
            raise TypeError("filament_scales_px must be a tuple.")
        if not self.filament_scales_px:
            raise ValueError("filament_scales_px cannot be empty.")
        for scale in self.filament_scales_px:
            _validate_positive_real(scale, "filament_scales_px")
        _validate_positive_integer(
            self.min_filament_length_px,
            "min_filament_length_px",
        )
        _validate_positive_real(self.bubble_edge_sigma, "bubble_edge_sigma")
        _validate_nonnegative_integer(
            self.bubble_closing_px,
            "bubble_closing_px",
        )
        _validate_positive_integer(
            self.min_bubble_area_px,
            "min_bubble_area_px",
        )
        _validate_fraction(
            self.min_bubble_boundary_fraction,
            "min_bubble_boundary_fraction",
        )


@dataclass(frozen=True)
class InternalStructureRegion:
    """One detected region, expressed in original-image coordinates."""

    label: int
    area_px: int
    centroid_yx: tuple[float, float]
    bbox_yx: tuple[int, int, int, int]
    mean_signed_residual: float
    structure_type: str = "unclassified"
    skeleton_length_px: int = 0

    @property
    def polarity(self) -> str:
        """Return the intensity polarity associated with this region type."""
        if self.structure_type == "light_region":
            return "bright"
        if self.structure_type in {"dark_filament", "bubble"}:
            return "dark"
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
        """Return the number of detected bubble regions."""
        return sum(region.structure_type == "bubble" for region in self.regions)

    def _area_fraction(self, mask: NDArray[np.bool_]) -> float:
        """Return mask area divided by usable interior area."""
        if self.usable_area_px == 0:
            return 0.0
        return float(np.count_nonzero(mask)) / self.usable_area_px

    def to_full_frame_mask(self) -> NDArray[np.bool_]:
        """Map the cropped structure mask back to the original image."""
        full_mask = np.zeros(self.original_shape, dtype=bool)
        y_start, x_start = self.crop_origin_yx
        y_stop = y_start + self.structure_mask.shape[0]
        x_stop = x_start + self.structure_mask.shape[1]
        full_mask[y_start:y_stop, x_start:x_stop] = self.structure_mask
        return full_mask


@dataclass(frozen=True)
class InternalStructureVideoSummary:
    """Framewise internal-structure abundance summarized over one video."""

    median_area_fraction: float
    upper_area_fraction: float
    frame_prevalence: float
    n_frames: int
    median_light_area_fraction: float = 0.0
    median_filament_area_fraction: float = 0.0
    median_filament_length_px: float = 0.0
    median_bubble_area_fraction: float = 0.0
    median_bubble_count: float = 0.0


def detect_internal_structures(
    frame: NDArray[np.generic],
    contour: ImageContour,
    config: InternalStructureConfig | None = None,
) -> InternalStructureFrameResult:
    """Detect light regions, dark filaments, and bubbles in one image frame.

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
        filament_mask = np.zeros_like(usable_mask)
        filament_skeleton = np.zeros_like(usable_mask)
        bubble_mask = np.zeros_like(usable_mask)
    else:
        normalized = np.zeros_like(residual)
        normalized[usable_mask] = residual[usable_mask] / noise_sigma
        light_mask = _detect_light_regions(normalized, usable_mask, settings)
        filament_mask, filament_skeleton = _detect_dark_filaments(
            normalized,
            usable_mask,
            settings,
        )
        bubble_mask = _detect_bubbles(normalized, usable_mask, settings)
        filament_mask &= ~bubble_mask
        filament_skeleton &= filament_mask

    structure_mask = light_mask | filament_mask | bubble_mask
    regions = (
        _describe_regions(
            label(light_mask, connectivity=2),
            residual,
            crop_origin,
            "light_region",
        )
        + _describe_regions(
            label(filament_mask, connectivity=2),
            residual,
            crop_origin,
            "dark_filament",
            filament_skeleton,
        )
        + _describe_regions(
            label(bubble_mask, connectivity=2),
            residual,
            crop_origin,
            "bubble",
        )
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
        dark_filament_mask=filament_mask,
        bubble_region_mask=bubble_mask,
        dark_filament_skeleton=filament_skeleton,
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
            np.median(
                [getattr(result, "light_area_fraction", 0.0) for result in results]
            )
        ),
        median_filament_area_fraction=float(
            np.median(
                [
                    getattr(result, "filament_area_fraction", 0.0)
                    for result in results
                ]
            )
        ),
        median_filament_length_px=float(
            np.median(
                [getattr(result, "filament_length_px", 0) for result in results]
            )
        ),
        median_bubble_area_fraction=float(
            np.median(
                [getattr(result, "bubble_area_fraction", 0.0) for result in results]
            )
        ),
        median_bubble_count=float(
            np.median(
                [getattr(result, "bubble_count", 0) for result in results]
            )
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
    return binary_erosion(
        interior_mask,
        footprint=disk(config.membrane_exclusion_px),
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
    usable_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> NDArray[np.bool_]:
    """Grow broad light regions outward from high-confidence positive seeds."""
    seeds = usable_mask & (
        normalized_residual >= config.threshold_sigma
    )
    candidates = usable_mask & (
        normalized_residual >= config.light_grow_sigma
    )
    grown = ndimage.binary_propagation(seeds, mask=candidates)
    return remove_small_objects(
        grown,
        min_size=config.min_region_area_px,
    )


def _dark_ridge_response(
    normalized_residual: NDArray[np.float64],
    scales: tuple[float, ...],
) -> NDArray[np.float64]:
    """Return scale-normalized positive Hessian response to dark ridges."""
    response = np.zeros_like(normalized_residual)
    for sigma in scales:
        hxx = ndimage.gaussian_filter(
            normalized_residual,
            sigma=sigma,
            order=(0, 2),
        )
        hyy = ndimage.gaussian_filter(
            normalized_residual,
            sigma=sigma,
            order=(2, 0),
        )
        hxy = ndimage.gaussian_filter(
            normalized_residual,
            sigma=sigma,
            order=(1, 1),
        )
        trace = hxx + hyy
        discriminant = np.sqrt((hxx - hyy) ** 2 + 4.0 * hxy ** 2)
        largest_eigenvalue = 0.5 * (trace + discriminant)
        response = np.maximum(
            response,
            np.maximum(largest_eigenvalue * sigma**2, 0.0),
        )
    return response


def _detect_dark_filaments(
    normalized_residual: NDArray[np.float64],
    usable_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Detect thin dark ridges and retain components with sufficient length."""
    ridge_response = _dark_ridge_response(
        normalized_residual,
        config.filament_scales_px,
    )
    candidates = usable_mask & (
        ridge_response >= config.filament_threshold_sigma
    )
    candidates &= normalized_residual < 0.0
    skeleton = skeletonize(candidates)
    kept_skeleton = np.zeros_like(skeleton)
    for component in regionprops(label(skeleton, connectivity=2)):
        if component.area >= config.min_filament_length_px:
            coordinates = component.coords
            kept_skeleton[coordinates[:, 0], coordinates[:, 1]] = True
    if not np.any(kept_skeleton):
        return np.zeros_like(candidates), kept_skeleton
    max_radius = int(np.ceil(max(config.filament_scales_px)))
    filament_mask = candidates & binary_dilation(
        kept_skeleton,
        footprint=disk(max_radius),
    )
    return filament_mask, skeletonize(filament_mask)


def _detect_bubbles(
    normalized_residual: NDArray[np.float64],
    usable_mask: NDArray[np.bool_],
    config: InternalStructureConfig,
) -> NDArray[np.bool_]:
    """Detect dark, sufficiently closed boundaries and fill their interiors."""
    dark_edge = usable_mask & (
        normalized_residual <= -config.bubble_edge_sigma
    )
    if config.bubble_closing_px:
        closed_edge = binary_closing(
            dark_edge,
            footprint=disk(config.bubble_closing_px),
        )
    else:
        closed_edge = dark_edge
    filled = ndimage.binary_fill_holes(closed_edge) & usable_mask
    enclosed = filled & ~closed_edge
    bubble_mask = np.zeros_like(usable_mask)
    for candidate in regionprops(label(enclosed, connectivity=2)):
        if candidate.area < config.min_bubble_area_px:
            continue
        interior = np.zeros_like(usable_mask)
        coordinates = candidate.coords
        interior[coordinates[:, 0], coordinates[:, 1]] = True
        boundary = binary_dilation(interior, footprint=disk(1)) & ~interior
        boundary &= usable_mask
        boundary_size = np.count_nonzero(boundary)
        if boundary_size == 0:
            continue
        coverage = np.count_nonzero(boundary & dark_edge) / boundary_size
        if coverage < config.min_bubble_boundary_fraction:
            continue
        bubble_mask |= interior | (boundary & closed_edge)
    return bubble_mask


def _describe_regions(
    labelled: NDArray[np.int_],
    residual: NDArray[np.float64],
    crop_origin: tuple[int, int],
    structure_type: str,
    skeleton: NDArray[np.bool_] | None = None,
) -> tuple[InternalStructureRegion, ...]:
    """Convert labelled crop regions to original-image measurements."""
    y_offset, x_offset = crop_origin
    descriptions = []
    for region in regionprops(labelled, intensity_image=residual):
        min_y, min_x, max_y, max_x = region.bbox
        centroid_y, centroid_x = region.centroid
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
                structure_type=structure_type,
                skeleton_length_px=(
                    0
                    if skeleton is None
                    else int(
                        np.count_nonzero(
                            skeleton[
                                min_y:max_y,
                                min_x:max_x,
                            ]
                            & region.image
                        )
                    )
                ),
            )
        )
    return tuple(descriptions)


def _validate_positive_real(value: object, name: str) -> None:
    """Require a finite positive real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _validate_nonnegative_integer(value: object, name: str) -> None:
    """Require a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_integer(value: object, name: str) -> None:
    """Require a positive integer."""
    _validate_nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive.")


def _validate_fraction(value: object, name: str) -> None:
    """Require a finite real number in the closed unit interval."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one.")
