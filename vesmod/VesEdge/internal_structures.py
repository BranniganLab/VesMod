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
from skimage.morphology import binary_erosion, disk, remove_small_objects

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
        Minimum absolute residual as a multiple of the robust residual noise.
    min_region_area_px : int
        Minimum connected-component area retained as structure, in pixels.
    """

    membrane_exclusion_px: int = 5
    background_sigma_px: float = 8.0
    threshold_sigma: float = 4.0
    min_region_area_px: int = 9

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


@dataclass(frozen=True)
class InternalStructureRegion:
    """One detected region, expressed in original-image coordinates."""

    label: int
    area_px: int
    centroid_yx: tuple[float, float]
    bbox_yx: tuple[int, int, int, int]
    mean_signed_residual: float

    @property
    def polarity(self) -> str:
        """Return whether the region is brighter or darker than background."""
        if self.mean_signed_residual >= 0:
            return "bright"
        return "dark"


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


def detect_internal_structures(
    frame: NDArray[np.generic],
    contour: ImageContour,
    config: InternalStructureConfig | None = None,
) -> InternalStructureFrameResult:
    """Detect bright and dark interior structure in one image frame.

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
        structure_mask = np.zeros_like(usable_mask)
    else:
        structure_mask = usable_mask & (
            np.abs(residual) >= settings.threshold_sigma * noise_sigma
        )
        structure_mask = remove_small_objects(
            structure_mask,
            min_size=settings.min_region_area_px,
        )

    labelled = label(structure_mask, connectivity=2)
    regions = _describe_regions(labelled, residual, crop_origin)
    return InternalStructureFrameResult(
        original_shape=frame.shape,
        crop_origin_yx=crop_origin,
        usable_interior_mask=usable_mask,
        residual=residual,
        structure_mask=structure_mask,
        regions=regions,
        noise_sigma=noise_sigma,
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


def _describe_regions(
    labelled: NDArray[np.int_],
    residual: NDArray[np.float64],
    crop_origin: tuple[int, int],
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
