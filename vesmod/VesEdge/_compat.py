"""Keep historical VesEdge module paths pointing at the same modules."""

from importlib import import_module
import sys


def register_legacy_modules() -> None:
    """Expose moved modules under their original import paths."""
    package = sys.modules[__package__]
    for old_name, new_name in _MOVED_MODULES.items():
        module = import_module(new_name, __package__)
        sys.modules[f"{__package__}.{old_name}"] = module
        setattr(package, old_name, module)


_MOVED_MODULES = {
    "edge_extractor": ".extraction.edge_extractor",
    "vesicle_video_utils": ".extraction.vesicle_video_utils",
    "checkpoint_io": ".io.checkpoint_io",
    "frame_source": ".io.frame_source",
    "qc_checks": ".qc.qc_checks",
    "qc_config": ".qc.qc_config",
    "qc_replay": ".qc.qc_replay",
    "area_qc": ".qc.checks.area_qc",
    "edge_filtering": ".qc.checks.edge_filtering",
    "localized_deviation_qc": ".qc.checks.localized_deviation_qc",
    "minimum_radius_qc": ".qc.checks.minimum_radius_qc",
    "singleton_qc": ".qc.checks.singleton_qc",
    "animation": ".visualization.animation",
    "trace_plotting": ".visualization.trace_plotting",
}
