"""Example extraction, checkpoint, and re-QC workflow for VesEdge.

Equivalent CLI workflow::

    vesedge extract ./videos --pixels-per-micron 13.44 \
        --downsample --n-samples 120 --output-dir ./checkpoints

    vesedge qc ./checkpoints --curvature-threshold 0.059 \
        --output-dir ./results/qc_standard

    vesedge qc ./checkpoints --curvature-threshold 0.089 \
        --output-dir ./results/qc_permissive
"""

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
    edges.save_checkpoint(path)

    # A GIF can be generated only while the original image frames are present.
    video.make_vesicle_gif(path, edges)


# Later, load the same checkpoint and evaluate it under any QC configuration.
qc_config = EdgeQCConfig(
    curvature_threshold=0.059,
)

# edges = VesicleEdges.from_checkpoint("YOUR/PATH/HERE/sample.npz")
# edges.run_qc(qc_config)
# print(edges.qc_result.curvature)
# edges.save_edge_to_npy("YOUR/PATH/HERE/results/qc_standard/sample.npy")

# Evaluate the same checkpoint again without rerunning extraction.
# permissive_qc = EdgeQCConfig(
#     curvature_threshold=0.089,
# )
# edges.run_qc(permissive_qc)
# edges.save_edge_to_npy("YOUR/PATH/HERE/results/qc_permissive/sample.npy")
