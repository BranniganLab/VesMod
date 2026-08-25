"""Contour-area deviation quality control for VesEdge trajectories."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .models import AreaQCResult, EdgeDetection, QCFlag


def contour_area(contour_radii: NDArray[np.float64]) -> float:
    """Return enclosed area for uniformly sampled radial coordinates.

    The polar-area integral is approximated as pi * mean(r**2). This preserves
    noncircular radial variation, unlike pi * mean(r)**2.

    Parameters
    ----------
    contour_radii : NDArray[np.float64]
        Radial contour coordinates sampled uniformly over [0, 2*pi).

    Returns
    -------
    float
        Enclosed area in squared pixel units. Non-finite input produces nan so
        the corresponding detection can be rejected by QC.

    Raises
    ------
    ValueError
        If the radii are not a nonempty one-dimensional array.
    """
    radii = np.asarray(contour_radii)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("contour_radii must be a nonempty 1D array.")
    if not np.all(np.isfinite(radii)):
        return float("nan")
    return float(np.pi * np.mean(np.square(radii)))


def check_area_deviation(
    detections: list[EdgeDetection],
    max_relative_deviation: float,
) -> AreaQCResult:
    """Flag contours that deviate from the trajectory median enclosed area.

    A detection fails only when its absolute fractional area deviation is
    strictly greater than max_relative_deviation. A detection exactly on
    either configured bound passes.

    Parameters
    ----------
    detections : list[EdgeDetection]
        Successful detections from one vesicle trajectory.
    max_relative_deviation : float
        Maximum allowed abs(area - median_area) / median_area.

    Returns
    -------
    AreaQCResult
        Areas, trajectory reference, deviations, and rejection count.

    Raises
    ------
    ValueError
        If no detections are supplied, the threshold is invalid, or no finite
        positive contour area is available as a reference.
    """
    if not detections:
        raise ValueError("Area QC requires at least one successful detection.")
    if (
        not np.isfinite(max_relative_deviation)
        or not 0 <= max_relative_deviation < 1
    ):
        raise ValueError(
            "max_relative_deviation must be finite, at least 0, and less than 1."
        )

    areas = np.asarray(
        [contour_area(edge.full_contour.r) for edge in detections],
        dtype=np.float64,
    )
    reference_candidates = areas[np.isfinite(areas) & (areas > 0)]
    if reference_candidates.size == 0:
        raise ValueError(
            "Area QC requires at least one finite positive contour area."
        )
    reference_area = float(np.median(reference_candidates))
    deviations = np.abs(areas - reference_area) / reference_area

    for edge, area, deviation in zip(
        detections,
        areas,
        deviations,
        strict=True,
    ):
        edge.qc.area_pixels2 = float(area)
        edge.qc.relative_area_deviation = float(deviation)
        exceeds_threshold = (
            deviation > max_relative_deviation
            and not np.isclose(
                deviation,
                max_relative_deviation,
                rtol=1e-12,
                atol=1e-15,
            )
        )
        if not np.isfinite(deviation) or exceeds_threshold:
            edge.qc.flags.add(QCFlag.AREA_DEVIATION)
        else:
            edge.qc.flags.discard(QCFlag.AREA_DEVIATION)

    rejected_count = sum(
        QCFlag.AREA_DEVIATION in edge.qc.flags
        for edge in detections
    )
    return AreaQCResult(
        areas_pixels2=tuple(float(area) for area in areas),
        reference_area_pixels2=reference_area,
        relative_deviations=tuple(
            float(deviation)
            for deviation in deviations
        ),
        rejected_count=rejected_count,
    )
