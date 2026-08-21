"""
Example usage for VesEdge.

Update fpath and pixels_per_micron and run this file, or adapt it for your own use.
"""
import glob
from pathlib import Path

import nd2

from vesmod.VesEdge import (
    EdgeExtractionConfig,
    EdgeQCConfig,
    VesicleVideo,
    extract_edge_from_frame,
)

fpath = "YOUR/PATH/HERE/"  # Directory containing your .nd2 file(s).
pixels_per_micron = 13.44  # Microscope image calibration.

extraction_config = EdgeExtractionConfig(
    pixels_per_micron=pixels_per_micron,
    n_angular_samples=120,
)
qc_config = EdgeQCConfig(
    curvature_threshold=10,
    population_bic_threshold=10,
    max_minor_population_fraction=0.25,
)

for file in glob.glob(fpath + "*.nd2", recursive=True):
    path = Path(file).resolve()
    print(f"working on file {path.stem}")

    if path.with_suffix(".gif").exists():
        # Skip this file because edge extraction was already performed.
        continue

    intensities = nd2.imread(path)
    video = VesicleVideo(
        intensities,
        extraction_config,
        qc_config,
    )
    video.extract_edges(extract_edge_from_frame)

    # Save all extraction results, including detections rejected by the current
    # QC settings, if you may want to rerun QC later without re-extracting.
    video.save_checkpoint(
        path.with_name(f"{path.stem}_checkpoint.npz")
    )

    video.make_vesicle_gif(path, show_trace=True)
    video.save_edge_to_npy(path)


# A checkpoint can later be loaded without the original image frames and
# reevaluated using different QC settings:
# new_qc_config = EdgeQCConfig(
#     curvature_threshold=8,
#     population_bic_threshold=10,
#     max_minor_population_fraction=0.25,
# )
# reloaded_video = VesicleVideo.from_checkpoint(
#     "YOUR/PATH/HERE/sample_checkpoint.npz",
#     qc_config=new_qc_config,
# )
# reloaded_video.save_edge_to_npy(
#     "YOUR/PATH/HERE/sample_reanalyzed.npy"
# )
