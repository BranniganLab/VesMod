"""Example extraction, checkpoint, and re-QC workflow for VesEdge.

Equivalent CLI workflow::

    vesedge extract ./videos --pixels-per-micron 13.44 \
        --n-angular-samples 120 --output-dir ./checkpoints

    vesedge qc ./checkpoints --curvature-threshold 0.059 \
        --output-dir ./results/qc_standard

    vesedge qc ./checkpoints --curvature-threshold 0.089 \
        --output-dir ./results/qc_permissive
"""

import glob
from pathlib import Path

import nd2

from vesmod.VesEdge import (
    CurvatureQCConfig,
    EdgeExtractionConfig,
    VesicleEdges,
    VesicleQCConfig,
    VesicleVideo,
    extract_edge_from_frame,
    make_vesicle_gif,
)

fpath = "YOUR/PATH/HERE/"  # Directory containing your .nd2 file(s).
pixels_per_micron = 13.44  # Microscope image calibration.

extraction_config = EdgeExtractionConfig(
    pixels_per_micron=pixels_per_micron,
    n_angular_samples=120,
    calibration_source="measured",
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

    # Rendering is separate from the video model and can combine frames with
    # the extracted edge results while the image data are available.
    make_vesicle_gif(video, path, edges)


# Later, load the same checkpoint and evaluate it under any QC configuration.
qc_config = VesicleQCConfig(
    curvature=CurvatureQCConfig(threshold=0.059),
)

# edges = VesicleEdges.from_checkpoint("YOUR/PATH/HERE/sample.npz")
# edges.run_qc(qc_config)
# print(edges.qc_result.curvature)
# edges.export_accepted_radii("YOUR/PATH/HERE/results/qc_standard/sample.npy")

# Evaluate the same checkpoint again without rerunning extraction.
# permissive_qc = VesicleQCConfig(
#     curvature=CurvatureQCConfig(threshold=0.089),
# )
# edges.run_qc(permissive_qc)
# edges.export_accepted_radii("YOUR/PATH/HERE/results/qc_permissive/sample.npy")
