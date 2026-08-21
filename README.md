# VesMod

VesMod is a Python package for extracting membrane contours from microscopy videos of giant unilamellar vesicles (GUVs) and estimating membrane bending rigidity from thermal shape fluctuations.

The package separates image-dependent edge extraction from quality control and downstream fluctuation analysis so extracted contours can be reused under multiple QC configurations.

---

## Components

### VesEdge

VesEdge extracts vesicle contours from microscopy images and videos.

Features include:

* Automated vesicle edge detection
* Processing of ND2 microscopy videos
* User-supplied edge extraction algorithms
* Reusable `.npz` extraction checkpoints
* Frame-level curvature quality control
* Trajectory-level center/radius population quality control
* Rerunnable QC without repeating edge extraction
* Optional angular downsampling
* NumPy export of accepted contours
* Annotated GIF generation while image frames are available

Detailed documentation:

```text
docs/VesEdge_CLI_README.md
```

### EdgeMod

EdgeMod analyzes accepted vesicle contours and estimates membrane mechanical parameters.

Features include:

* Fourier analysis of contour fluctuations
* Estimation of membrane bending modulus (kC)
* Optional estimation of membrane tension (σ)
* JSON export of analysis results
* Batch processing of contour datasets

Detailed documentation:

```text
docs/EdgeMod_CLI_README.md
```

---

## Installation

```bash
git clone https://github.com/BranniganLab/VesMod.git
cd VesMod
pip install .
```

Verify the command-line tools:

```bash
vesedge --help
edgemod --help
```

---

## Recommended Workflow

### 1. Extract once and save checkpoints

```bash
vesedge ./videos --pixels-per-micron 13.44 --downsample --n-samples 120
```

For each ND2 file, VesEdge extracts contours and writes a reusable checkpoint:

```text
sample.npz
sample.gif
```

The `.npz` file is the persistent extraction product. It contains successful detections and extraction failures, but it does **not** contain QC decisions. This allows the same extraction results to be evaluated repeatedly without rerunning image processing.

The Python data model reflects this separation:

```python
from vesmod.VesEdge import (
    EdgeExtractionConfig,
    VesicleVideo,
    extract_edge_from_frame,
)

video = VesicleVideo(frames)
edges = video.extract_edges(
    extract_edge_from_frame,
    EdgeExtractionConfig(
        pixels_per_micron=13.44,
        n_angular_samples=120,
    ),
)
edges.save_checkpoint("sample.npz")
```

`VesicleVideo` owns raw image frames and image-dependent operations. `VesicleEdges` owns extracted contours, quality control, checkpoint persistence, and accepted-contour export.

### 2. Evaluate a QC configuration

```python
from vesmod.VesEdge import EdgeQCConfig, VesicleEdges

edges = VesicleEdges.from_checkpoint("sample.npz")

qc_config = EdgeQCConfig(
    curvature_threshold=10.0,
    population_bic_threshold=10.0,
    max_minor_population_fraction=0.25,
)

edges.run_qc(qc_config)
edges.save_edge_to_npy("qc_standard/sample.npy")
```

Loading a checkpoint does not run QC automatically. `run_qc()` clears any existing derived QC state before applying the requested configuration.

The same checkpoint can then be evaluated under another configuration:

```python
permissive_qc = EdgeQCConfig(
    curvature_threshold=15.0,
    population_bic_threshold=10.0,
    max_minor_population_fraction=0.25,
)

edges.run_qc(permissive_qc)
edges.save_edge_to_npy("qc_permissive/sample.npy")
```

The `.npy` output contains only contours accepted by the current QC configuration and is the input expected by the EdgeMod CLI.

### 3. Analyze accepted contours

```bash
edgemod qc_standard
edgemod qc_permissive
```

Each `.npy` input produces a corresponding `.json` result containing fitted membrane mechanical parameters.

---

## Data Products

| File | Meaning |
| --- | --- |
| `.npz` | Reusable, QC-independent VesEdge extraction checkpoint |
| `.npy` | Accepted contour radii for one specific QC configuration |
| `.gif` | Visual inspection of image frames and extracted contours |
| `.json` | EdgeMod spectrum/fitting output |

---

## Documentation

| Component | Documentation |
| --- | --- |
| VesEdge | `docs/VesEdge_CLI_README.md` |
| EdgeMod | `docs/EdgeMod_CLI_README.md` |

---

## Contributing

Issues and pull requests are welcome. Please include a clear problem description, reproduction steps, relevant input files when possible, error messages, and tests for new functionality.

## License

VesMod is distributed under the GNU General Public License v3.0 (or, at your option, any later version). See the LICENSE file for details.

## Citation

If VesMod contributes to a publication, please cite the associated manuscript and software repository.
