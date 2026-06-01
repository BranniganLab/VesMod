from pathlib import Path
import glob
import nd2
import numpy as np
from vesmod.VesEdge import VesicleVideo, extract_edge_from_frame


for file in glob.glob(YOUR_PATH_HERE+'*.nd2', recursive=True):
    path = Path(file).resolve()
    print(f"working on file {path.stem}")
    if path.with_suffix(".gif").exists():
        # skip this file because edge extraction already performed
        continue
    intensities = nd2.imread(path)
    video = VesicleVideo(intensities)
    video.extract_edges(extract_edge_from_frame, curvature_threshold=5)
    video.make_vesicle_gif(path, True)
    video.save_edge_to_npy(path)
