# VesMod

VesMod is a Python package for extracting membrane contours from microscopy videos of giant unilamellar vesicles (GUVs) and estimating membrane bending rigidity from thermal shape fluctuations.

VesEdge separates image-dependent edge extraction from quality control so the same extracted contours can be evaluated under multiple QC configurations without rerunning image processing. EdgeMod separates the measured fluctuation spectrum from the physical-fit configuration used to analyze that spectrum. Experimental analysis methods are kept outside the stable core API.

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
* Trajectory-relative contour-area quality control
* Rerunnable QC without repeating edge extraction
* QC provenance and batch-summary outputs
* Optional angular downsampling
* NumPy export of accepted contours
* Annotated GIF generation for visual inspection

See the [VesEdge CLI guide](docs/VesEdge_CLI_README.md).

### EdgeMod

EdgeMod analyzes accepted vesicle contours and estimates membrane mechanical parameters.

Stable core features include:

* Fourier analysis of contour fluctuations
* Estimation of membrane bending modulus (kC)
* Optional estimation of membrane tension (σ)
* Config-driven physical fitting through `SpectrumFitConfig`
* Fixed Fourier-mode fitting with user-defined lower/upper q bounds
* Retention of multiple physical-fit results on the same `Spectrum`
* JSON export of fit values, q bounds, and physical-fit configuration
* Batch processing of contour datasets

Experimental EdgeMod features live under `vesmod.EdgeMod.experimental`. These currently include optional q^-3-based dynamic range selection and temporal-RMS screening. Experimental APIs may change as the methods are evaluated and are not part of the stable core EdgeMod interface.

See the [EdgeMod CLI guide](docs/EdgeMod_CLI_README.md).

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

## Recommended CLI Workflow

VesEdge uses two explicit stages:

```text
ND2 videos
   │
   │ vesedge extract
   ▼
QC-independent .npz checkpoints
   │
   ├── vesedge qc, configuration A ──▶ accepted .npy files
   ├── vesedge qc, configuration B ──▶ accepted .npy files
   └── vesedge qc, configuration C ──▶ accepted .npy files
                                            │
                                            ▼
                                         EdgeMod
```

Extract each microscopy video once and keep the resulting checkpoint. QC can then be rerun repeatedly without invoking the edge extractor again.

### 1. Extract videos to checkpoints

```bash
vesedge extract ./videos \
    --pixels-per-micron 13.44 \
    --downsample \
    --n-samples 120 \
    --output-dir ./checkpoints
```

This creates one `.npz` checkpoint per input video and, unless `--no-gif` is used, a GIF showing the extracted contours.

```text
checkpoints/
├── sample01.npz
├── sample01.gif
├── sample02.npz
└── sample02.gif
```

The checkpoint contains extraction state only. It does not store QC decisions.

### 2. Apply curvature QC

Use a dedicated output directory for each QC configuration:

```bash
vesedge qc ./checkpoints \
    --curvature-threshold 5 \
    --output-dir ./results/qc_standard
```

This creates:

```text
results/qc_standard/
├── sample01.npy
├── sample02.npy
├── sample01.area_qc.csv
├── sample01.area_qc.png
├── sample02.area_qc.csv
├── sample02.area_qc.png
├── vesedge_qc.json
└── qc_summary.csv
```

`vesedge_qc.json` records the exact QC configuration and input selection. `qc_summary.csv` reports extraction failures, curvature rejections, area-deviation rejections, accepted-frame counts, and accepted fractions for each checkpoint. When area QC produces `qc_result.area`, each checkpoint also receives an `*.area_qc.csv` file containing its per-frame area measurements and an `*.area_qc.png` area-versus-frame diagnostic plot. These two files are not generated when area QC is disabled with `--no-area-qc`.

### 3. Compare alternate QC configurations

For example:

```bash
vesedge qc ./checkpoints \
    --curvature-threshold 10 \
    --output-dir ./results/qc_permissive
```

Keeping each QC configuration in its own directory makes the analysis provenance explicit and allows the outputs to be compared directly.

Do not store unrelated `.npy` files in a QC output directory: an incompatible rerun with `--overwrite` clears NumPy arrays beneath that directory before writing the new result.

### 4. Analyze each QC result with EdgeMod

The default EdgeMod CLI uses the historical fixed q range, q = 3–7:

```bash
edgemod ./results/qc_standard
edgemod ./results/qc_permissive
```

Experimental dynamic q-range selection can be enabled explicitly. The selector runs before the stable physical fitter, searches only inside the configured candidate interval, and requires explicit acceptance thresholds:

```bash
edgemod ./results/qc_standard \
    --dynamic-range \
    --lower-fitting-bound 3 \
    --upper-fitting-bound 20 \
    --min-modes 5 \
    --slope-tolerance 0.2 \
    --max-log-rmse 0.1
```

Fixed output is written as `<input>.json`; experimental dynamic output is written as `<input>.dynamic.json`, so the two analyses can be run on the same contour files without overwriting one another.

---

## Equivalent Python Workflow

The Python API exposes the same separation between extraction, QC, stable physical fitting, and experimental analysis.

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

edges.save_checkpoint("sample.npz")
```

Reload the checkpoint and apply QC later:

```python
edges = VesicleEdges.from_checkpoint("sample.npz")

qc_config = EdgeQCConfig(
    curvature_threshold=10.0,
)

edges.run_qc(qc_config)
edges.save_edge_to_npy("sample.npy")
```

After a completed QC run, `edges.qc_result` contains the curvature configuration and aggregate result. Per-detection curvature annotations remain available on each `EdgeDetection.qc`.

Fit the accepted contours with the stable core EdgeMod API:

```python
from vesmod.EdgeMod import Spectrum, SpectrumFitConfig

spectrum = Spectrum("sample.npy")

fixed_config = SpectrumFitConfig(
    lower_bound=3,
    upper_bound=8,
)
fixed_fit = spectrum.extract_kc_from_fit(fixed_config)
```

Experimental dynamic range selection is an upstream operation that produces q bounds for the same core fitter:

```python
from dataclasses import replace
from vesmod.EdgeMod.experimental import QMinusThreeRangeSelector

selector = QMinusThreeRangeSelector(
    lower_bound=3,
    upper_bound=20,
    min_modes=5,
    slope_tolerance=0.2,
    max_log_rmse=0.1,
)
selection = selector.select(spectrum.modes, spectrum.avg_amps2)

if selection.accepted:
    dynamic_config = replace(
        fixed_config,
        lower_bound=selection.lower_bound,
        upper_bound=selection.upper_bound,
    )
    dynamic_fit = spectrum.extract_kc_from_fit(dynamic_config)
```

Both successful physical fits remain available in `spectrum.fit_results`. Each `SpectrumFit` records the actual q bounds used and the full core `SpectrumFitConfig`. Experimental selection diagnostics remain separate from the core `Spectrum`/`SpectrumFit` state.

---

## Data Products

| File | Meaning |
| --- | --- |
| `.npz` | Reusable, QC-independent VesEdge extraction checkpoint |
| `.gif` | Visual inspection of raw image frames with extracted contours |
| `.npy` | Accepted contour radii for one QC configuration, ready for EdgeMod |
| `vesedge_qc.json` | QC configuration and source path for one QC batch |
| `qc_summary.csv` | Per-video QC counts and accepted fractions for one QC batch |
| `.spectrum_diagnostic.png` | Measured spectrum, attempted fit, compensated spectrum, and fit residuals |
| `.json` | EdgeMod fixed-range spectrum/fitting output |
| `.dynamic.json` | EdgeMod output produced when experimental dynamic selection is requested |
| `temporal_rms_summary.csv` | Experimental temporal-RMS measurement and inclusion decision for each trajectory |
| `temporal_rms_histogram.png` | Experimental distribution of temporal-RMS amplitudes |
| `temporal_rms_qc.json` | Experimental temporal-RMS configuration, input manifest, and export manifest |

---

## Documentation

| Component | Documentation |
| --- | --- |
| VesEdge | [VesEdge CLI guide](docs/VesEdge_CLI_README.md) |
| EdgeMod | [EdgeMod CLI guide](docs/EdgeMod_CLI_README.md) |

---

## Contributing

Issues and pull requests are welcome. Please include a clear problem description, reproduction steps, relevant input files when possible, error messages, and tests for new functionality.

## License

VesMod is distributed under the GNU General Public License v3.0 (or, at your option, any later version). See the LICENSE file for details.

## Citation

If VesMod contributes to a publication, please cite the associated manuscript and software repository.
