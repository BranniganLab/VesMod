# VesMod

VesMod is a Python package for extracting membrane contours from microscopy videos of giant unilamellar vesicles (GUVs) and estimating membrane bending rigidity from thermal shape fluctuations.

The Python API separates image-dependent edge extraction from quality control and downstream fluctuation analysis so extracted contours can be reused under multiple QC configurations.

---

## Components

### VesEdge

VesEdge extracts vesicle contours from microscopy images and videos.

Features include:

* Automated vesicle edge detection
* Processing of ND2 microscopy videos
* User-supplied edge extraction algorithms
* Frame-level curvature quality control
* Trajectory-level center/radius population quality control
* Reusable `.npz` extraction checkpoints through the Python API
* Rerunnable QC without repeating edge extraction
* Optional angular downsampling
* NumPy export of accepted contours
* Annotated GIF generation for visual inspection

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

## Typical CLI Workflow

### 1. Extract, QC, and save accepted contours

```bash
vesedge sample.nd2 \
    --pixels-per-micron 13.44 \
    --downsample \
    --n-samples 120
```

This generates:

```text
sample.npy
sample.gif
```

The `.npy` file contains only contours accepted by the configured VesEdge QC checks and is ready for EdgeMod.

### 2. Estimate membrane mechanical properties

```bash
edgemod sample.npy
```

This generates:

```text
sample.json
```

containing fitted membrane mechanical parameters.

---

## Reusing Extraction Results in Python

The refactored API separates raw video data from extracted-edge state:

```python
from vesmod.VesEdge import (
    EdgeExtractionConfig,
    EdgeQCConfig,
    VesicleEdges,
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
```

Save QC-independent extraction state:

```python
edges.save_checkpoint("sample.npz")
```

Reload it later and apply a QC configuration without rerunning extraction:

```python
edges = VesicleEdges.from_checkpoint("sample.npz")

qc_config = EdgeQCConfig(
    curvature_threshold=10.0,
    population_bic_threshold=10.0,
    max_minor_population_fraction=0.25,
)

edges.run_qc(qc_config)
edges.save_edge_to_npy("sample.npy")
```

After a completed QC run, `edges.qc_result` contains the configuration and the results from each enabled QC stage. `edges.qc_result.curvature` summarizes curvature scores and rejections, while `edges.qc_result.population` contains the center/radius population-analysis result. Per-detection QC annotations remain available on each `EdgeDetection.qc`.

The same `VesicleEdges` object can be re-QCed with different settings. Checkpoints store extraction state only; QC decisions and `qc_result` are derived and reset on each `run_qc()` call.

---

## Data Products

| File | Meaning |
| --- | --- |
| `.npz` | Reusable, QC-independent VesEdge extraction checkpoint created through the Python API |
| `.npy` | Accepted contour radii for one QC configuration |
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
