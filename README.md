# VesMod

VesMod is a Python package for extracting membrane contours from microscopy videos of giant unilamellar vesicles (GUVs) and estimating membrane bending rigidity from thermal shape fluctuations.

VesEdge separates image-dependent edge extraction from quality control so the same extracted contours can be evaluated under multiple QC configurations without rerunning image processing. EdgeMod separates the measured fluctuation spectrum from the physical-fit configuration used to analyze that spectrum.

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
* Explicit angular sampling
* NumPy export of all accepted contour radii
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

`Spectrum` can calculate either time-averaged mean-square Fourier power or temporal variance around each mode's complex time mean.

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
   ├── vesedge gif ──────────────────────────▶ inspection GIFs
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
vesedge extract "./videos" \
    --pixels-per-micron 13.44 \
    --n-angular-samples 120 \
    --output-dir ./checkpoints
```

Calibration is required explicitly. For intentionally uncalibrated or dimensionless workflows, use `--assume-one-pixel-per-micron`. Angular sampling defaults to 120, matching the Python API; use `--n-angular-samples native` to retain the extractor's native sampling.

This creates one reusable `.npz` checkpoint per input video:

```text
checkpoints/
├── sample01.npz
└── sample02.npz
```

The checkpoint contains extraction state only. It does not store QC decisions.

### 2. Create inspection GIFs

GIF generation is an explicit post-extraction step. Render the source frames,
the extracted edges, or QC-colored edges into separate output directories:

```bash
vesedge gif "./checkpoints" --output-dir ./gifs/original --style original
vesedge gif "./checkpoints" --output-dir ./gifs/edges --style edges
vesedge gif "./checkpoints" \
    --output-dir ./gifs/qc_standard \
    --style qc \
    --qc-dir ./results/qc_standard
```

The `qc` style requires a completed `vesedge qc` run and its output directory
via `--qc-dir`; run that command after the next step. Add `--recursive` when the
checkpoint input contains nested directories.

### 3. Apply contour quality control

Use a dedicated output directory for each QC configuration:

```bash
vesedge qc "./checkpoints" \
    --curvature-threshold 21.520619405632544 \
    --localized-deviation-qc \
    --localized-deviation-order 3 \
    --max-localized-deviation-residual-fraction 0.05 \
    --output-dir ./results/qc_standard
```

This creates:

```text
results/qc_standard/
├── sample01.npy
├── sample02.npy
├── vesedge_qc.json
└── qc_summary.csv
```

`vesedge_qc.json` records the QC settings and input selection. Curvature and localized-deviation checks run by default. `qc_summary.csv` reports extraction failures, rejections, and accepted-frame counts and fractions.

### 4. Compare alternate QC configurations

For example:

```bash
vesedge qc "./checkpoints" \
    --curvature-threshold 32.46298468904425 \
    --localized-deviation-qc \
    --localized-deviation-order 3 \
    --max-localized-deviation-residual-fraction 0.05 \
    --output-dir ./results/qc_permissive
```

Keeping each QC configuration in its own directory makes the analysis provenance explicit and allows the outputs to be compared directly.

QC records its generated files in the output manifest. An incompatible rerun with
`--overwrite` removes only those recorded files; unrelated arrays and other files
in the directory are preserved.

### 5. Analyze each QC result with EdgeMod

The default EdgeMod CLI uses the historical fixed q range, q = 3–7:

```bash
edgemod "./results/qc_standard"
edgemod "./results/qc_permissive"
```

Fit output is written as `<input>.json`.

---

## Equivalent Python Workflow

The Python API provides the same extraction, QC, and fitting workflow.

```python
from pathlib import Path

from vesmod.VesEdge.extraction.config import EdgeExtractionConfig
from vesmod.VesEdge.vesicle_video import VesicleVideo
from vesmod.VesEdge.extraction.edge_extractor import extract_edge_from_frame

video = VesicleVideo(frames)
edges = video.extract_edges(
    extract_edge_from_frame,
    EdgeExtractionConfig(
        pixels_per_micron=13.44,
        n_angular_samples=120,
        calibration_source="measured",
    ),
)

Path("checkpoints").mkdir(exist_ok=True)
edges.save_checkpoint("checkpoints/sample.npz")
```

Reload the checkpoint and apply the same two QC configurations used in the CLI workflow.

```python
from vesmod.VesEdge.qc.checks.edge_filtering import CurvatureQCConfig
from vesmod.VesEdge.qc.checks.localized_deviation_qc import LocalizedDeviationQCConfig
from vesmod.VesEdge.qc.qc_config import VesicleQCConfig
from vesmod.VesEdge.vesicle_edges import VesicleEdges

edges = VesicleEdges.from_checkpoint("checkpoints/sample.npz")
localized_deviation = LocalizedDeviationQCConfig(
    enabled=True, order=3, max_residual_fraction=0.05,
)

qc_config = VesicleQCConfig(
    curvature=CurvatureQCConfig(threshold=21.520619405632544, enabled=True),
    localized_deviation=localized_deviation,
)
edges.run_qc(qc_config)
edges.export_accepted_radii("sample.npy")

permissive_qc = VesicleQCConfig(
    curvature=CurvatureQCConfig(threshold=32.46298468904425, enabled=True),
    localized_deviation=localized_deviation,
)
edges.run_qc(permissive_qc)
edges.export_accepted_radii("sample_permissive.npy")
```

After a completed QC run, `edges.qc_result` contains the aggregate result for the `VesicleQCConfig` that was applied. `edges.qc_result.rejected_by(frame_index)` reports the checks that rejected a source frame, and `diagnostics_for_frame(frame_index)` returns its measurements grouped by check. The final selection is available as `accepted_frame_indices`; see the [QC guide](docs/QC_checks.md#combining-decisions) to combine failures with an explicit rejection rule. `export_accepted_radii()` writes the analysis-contour radii from all accepted detections, converted to microns, as the `.npy` handoff to EdgeMod.

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

Successful fits remain available in `spectrum.fit_results`. Each `SpectrumFit` records the q bounds used and its `SpectrumFitConfig`.

Fit a replica-averaged spectrum with the same configuration object. Ensemble
fits use free sigma by default, including the `.kC` property:

```python
from vesmod.EdgeMod import SpectrumEnsemble, SpectrumFitConfig

ensemble = SpectrumEnsemble()
# Add compatible replica spectra with ensemble.add_spectrum(...).
ensemble_fit = ensemble.extract_kc_from_fit()

print(ensemble_fit.kC, ensemble_fit.reduced_sigma)

# Optionally fix reduced surface tension to zero.
fixed_sigma_fit = ensemble.extract_kc_from_fit(SpectrumFitConfig(free_sigma=False))
```

`EnsembleFit.reduced_sigma` is dimensionless. An ensemble has no single radius,
so VesMod does not convert it to SI surface tension.

---

## Data Products

| File | Meaning |
| --- | --- |
| `.npz` | Reusable, QC-independent VesEdge extraction checkpoint |
| `.gif` | Visual inspection created explicitly by `vesedge gif` in original, edge-overlay, or QC-colored style |
| `.npy` | All QC-accepted analysis-contour radii in microns for one QC configuration, ready for EdgeMod |
| `vesedge_qc.json` | QC configuration and source path for one QC batch |
| `qc_summary.csv` | Per-video QC counts and accepted fractions for one QC batch |
| `.spectrum_diagnostic.png` | Measured spectrum, attempted fit, compensated spectrum, and fit residuals |
| `.json` | EdgeMod fixed-range spectrum/fitting output |

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

See [replaceable power calculators and migration](docs/EdgeMod_power_calculators.md) for the calculator contract and `Spectrum.powers` API.
