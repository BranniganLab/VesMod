"""Example extraction, checkpoint, and re-QC workflow for VesEdge."""

import glob
from pathlib import Path

import nd2

from vesmod.VesEdge import (
    EdgeExtractionConfig,
    EdgeQCConfig,
    VesicleEdges,
    VesicleVideo,
    extract_edge_from_frame,
)

fpath = "YOUR/PATH/HERE/"  # Directory containing your .nd2 file(s).
pixels_per_micron = 13.44  # Microscope image calibration.

extraction_config = EdgeExtractionConfig(
    pixels_per_micron=pixels_per_micron,
    n_angular_samples=120,
)

for file in glob.glob(fpath + "*.nd2", recursive=True):
    path = Path(file).resolve()
    print(f"working on file {path.stem}")

    intensities = nd2.imread(path)
    video = VesicleVideo(intensities)
    edges = video.extract_edges(
        extract_edge_from_frame,
        extraction_config,
    )

    # The checkpoint is the reusable output of extraction. It stores all
    # successful detections and extraction failures, but no QC decisions.
    # Contours remain in image-space pixels; pixels_per_micron is stored with
    # the checkpoint so physical radii can be derived later.
    edges.save_checkpoint(path)

    # A GIF can be generated while the original image frames are available.
    video.make_vesicle_gif(path, edges)


# Later, load a checkpoint and evaluate it under any QC configuration.
qc_config = EdgeQCConfig(
    curvature_threshold=10,
    population_bic_threshold=10,
    max_minor_population_fraction=0.25,
)

# edges = VesicleEdges.from_checkpoint("YOUR/PATH/HERE/sample.npz")
# edges.run_qc(qc_config)
# edges.save_edge_to_npy("YOUR/PATH/HERE/qc_standard/sample.npy")

# The same checkpoint can be evaluated again without rerunning extraction.
# permissive_qc = EdgeQCConfig(
#     curvature_threshold=15,
#     population_bic_threshold=10,
#     max_minor_population_fraction=0.25,
# )
# edges.run_qc(permissive_qc)
# edges.save_edge_to_npy("YOUR/PATH/HERE/qc_permissive/sample.npy")
