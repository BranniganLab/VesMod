"""Public VesEdge API."""

from .visualization.animation import (
    AnimationPanel,
    TimeSeriesAnimationPanel,
    VesicleAnimationPanel,
    draw_vesicle_frame,
    make_gif,
    make_vesicle_gif,
)
from .extraction.config import EdgeExtractionConfig
from .extraction.edge_extractor import extract_edge_from_frame
from .qc.checks.edge_filtering import CurvatureQCConfig, CurvatureQCResult
from .qc.checks.localized_deviation_qc import LocalizedDeviationQCConfig, check_localized_deviation
from .contour_geometry import RadialBaselineFit, contour_area, fit_radial_baseline
from .io.frame_source import (
    OnDemandFrameSequence,
    as_frame_source,
    open_frame_source,
)
from .models import EdgeDetection, EdgeDetectionFailure, EdgeResult, ImageContour, VesicleQCResult
from .vesicle_edges import VesicleEdges
from .vesicle_video import VesicleVideo
from .qc.qc_config import VesicleQCConfig
from .qc.qc_replay import (
    QCReplayResult,
    RecordedQCSelection,
    load_recorded_qc,
    replay_recorded_qc,
)
from .visualization.trace_plotting import (
    EdgeTracePlotConfig,
    centered_contour_coordinates,
    contour_center_of_mass,
    plot_edge_traces,
    select_trace_detections,
)

__all__ = [
    "AnimationPanel",
    "CurvatureQCConfig",
    "CurvatureQCResult",
    "EdgeDetection",
    "EdgeDetectionFailure",
    "EdgeExtractionConfig",
    "EdgeResult",
    "EdgeTracePlotConfig",
    "ImageContour",
    "LocalizedDeviationQCConfig",
    "OnDemandFrameSequence",
    "QCReplayResult",
    "RadialBaselineFit",
    "RecordedQCSelection",
    "TimeSeriesAnimationPanel",
    "VesicleAnimationPanel",
    "VesicleEdges",
    "VesicleQCConfig",
    "VesicleQCResult",
    "VesicleVideo",
    "as_frame_source",
    "centered_contour_coordinates",
    "check_localized_deviation",
    "contour_area",
    "contour_center_of_mass",
    "draw_vesicle_frame",
    "extract_edge_from_frame",
    "fit_radial_baseline",
    "load_recorded_qc",
    "make_gif",
    "make_vesicle_gif",
    "open_frame_source",
    "plot_edge_traces",
    "replay_recorded_qc",
    "select_trace_detections",
]
