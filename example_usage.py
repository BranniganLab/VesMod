from pathlib import Path
import glob
import nd2
import numpy as np
from vesicle_edge_extractor.vesicle_video import VesicleVideo
from vesicle_edge_extractor.edge_extractor import extract_edge_from_frame


for file in glob.glob('/home/js2746/DOPC_Cer*/CPG*/**/*.nd2', recursive=True):
    path = Path(file).resolve()
    print(f"working on file {path.stem}")
    if path.with_suffix(".gif").exists():
        continue
    intensities = nd2.imread(path)
    video = VesicleVideo(intensities)
    video.extract_edges(extract_edge_from_frame, curvature_threshold=5)
    video.make_vesicle_gif(path, True)
    np.save(path.with_suffix(".npy"), video.r_vals)