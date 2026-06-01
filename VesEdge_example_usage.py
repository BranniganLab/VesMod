"""
Example usage for VesEdge.

Update fpath and pixel_to_micron_ratio and run this file, or adapt for your own use.
"""
from pathlib import Path
import glob
import nd2
from vesmod.VesEdge import VesicleVideo, extract_edge_from_frame

fpath = "YOUR PATH HERE"  # path to the directory containing your .nd2 file(s)
pixel_to_micron_ratio = "YOUR RATIO HERE"  # How many pixels to 1 micron in your micrscope image?

for file in glob.glob(fpath + '*.nd2', recursive=True):
    path = Path(file).resolve()
    print(f"working on file {path.stem}")
    if path.with_suffix(".gif").exists():
        # skip this file because edge extraction already performed
        continue
    intensities = nd2.imread(path)
    video = VesicleVideo(intensities, pixel_to_micron_ratio)
    video.extract_edges(extract_edge_from_frame, curvature_threshold=10)
    video.make_vesicle_gif(path, show_trace=True)
    video.save_edge_to_npy(path)
