# VesEdge CLI

The VesEdge CLI performs the image-dependent stage of the workflow: it extracts vesicle contours from ND2 microscopy videos and saves reusable `.npz` checkpoints. Quality control is deliberately separate so the same extracted contours can later be evaluated under multiple QC configurations without repeating edge extraction.

## Quick Start

```bash
vesedge sample.nd2 --pixels-per-micron 13.44 --downsample --n-samples 120
```

This writes:

```text
sample.npz
sample.gif
```

The `.npz` checkpoint is the persistent extraction product. The GIF is an optional visual preview of the extracted contours over the source images.

---

## Input Selection

Process one ND2 file:

```bash
vesedge sample.nd2
```

Process all ND2 files in a directory:

```bash
vesedge ./videos
```

Search subdirectories recursively:

```bash
vesedge ./videos --recursive
```

---

## Microscope Calibration

### `--pixels-per-micron`

```bash
vesedge sample.nd2 --pixels-per-micron 13.44
```

Specifies the microscope calibration in pixels per micron. The checkpoint stores both the image-space contours and analysis radii converted to microns.

Default: `1.0`.

---

## Angular Sampling

### `--downsample`

```bash
vesedge sample.nd2 --downsample
```

Resamples successful contours to a fixed angular grid before storing the analysis contour.

### `--n-samples`

```bash
vesedge sample.nd2 --downsample --n-samples 120
```

Sets the number of angular samples retained when downsampling is enabled.

Default: `120`.

Without `--downsample`, the native angular sampling returned by the extractor is retained. Successful detections must then have consistent analysis-contour lengths.

---

## Extraction Algorithm

By default VesEdge uses:

```text
vesmod.VesEdge:extract_edge_from_frame
```

### Importable custom extractor

```bash
vesedge sample.nd2 --extractor my_package.my_module:my_edge_extractor
```

### Extractor from a Python file

```bash
vesedge sample.nd2 \
    --extractor-file ./my_extractor.py \
    --extractor-name my_edge_extractor
```

See `custom_edge_extraction_algorithms.README.md` for the required function interface.

---

## GIF Output

By default, VesEdge writes a GIF with successfully extracted contours overlaid on the original frames.

Disable it with:

```bash
vesedge sample.nd2 --no-gif
```

The extraction CLI does not run QC, so the GIF is an extraction preview rather than a QC acceptance/rejection visualization.

---

## Existing Outputs

By default an input is skipped if an expected `.npz` checkpoint or GIF already exists.

Force re-extraction with:

```bash
vesedge sample.nd2 --overwrite
```

---

## Checkpoint Contents

A VesEdge `.npz` checkpoint stores the information required to reconstruct `VesicleEdges` without the original image frames:

- the extraction configuration;
- successful detections;
- extraction failures and their frame ordering;
- image-space contour origins;
- native extracted contours;
- analysis contours;
- analysis radii in microns.

QC settings and QC results are intentionally **not** stored. The checkpoint represents extraction state, not one particular interpretation of that state.

Raw image frames are also not stored in the checkpoint.

---

## Python Data Model

`VesicleVideo` owns raw image data and image-dependent operations:

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
```

`extract_edges()` returns a separate `VesicleEdges` object. It does not run QC.

`VesicleEdges` owns extracted results, quality control, checkpoint persistence, and accepted-contour export:

```python
edges.save_checkpoint("sample.npz")
```

Reload later with:

```python
from vesmod.VesEdge import VesicleEdges

edges = VesicleEdges.from_checkpoint("sample.npz")
```

Loading a checkpoint does not run QC automatically.

---

## Running Quality Control

Create a QC configuration and apply it explicitly:

```python
from vesmod.VesEdge import EdgeQCConfig

qc_config = EdgeQCConfig(
    curvature_threshold=10.0,
    population_bic_threshold=10.0,
    max_minor_population_fraction=0.25,
)

edges.run_qc(qc_config)
edges.save_edge_to_npy("qc_standard/sample.npy")
```

Calling `run_qc()` clears prior derived QC state before applying the new configuration. The same `VesicleEdges` object can therefore be evaluated repeatedly:

```python
permissive_qc = EdgeQCConfig(
    curvature_threshold=15.0,
    population_bic_threshold=10.0,
    max_minor_population_fraction=0.25,
)

edges.run_qc(permissive_qc)
edges.save_edge_to_npy("qc_permissive/sample.npy")
```

A `.npy` file is a filtered analysis product for one QC configuration. It contains only accepted radii and can be passed directly to EdgeMod.

---

## Recommended Analysis Pattern

```text
ND2 files
   │
   │ vesedge: extract once
   ▼
NPZ checkpoints
   │
   ├── QC configuration A ──▶ NPY files ──▶ EdgeMod
   ├── QC configuration B ──▶ NPY files ──▶ EdgeMod
   └── QC configuration C ──▶ NPY files ──▶ EdgeMod
```

Keep checkpoints as the reusable extraction record. Store `.npy` outputs for different QC configurations in separate directories so each downstream EdgeMod result can be traced to the QC settings that produced it.

---

## Troubleshooting

### No successful detections

If extraction fails on every frame, VesEdge cannot create a checkpoint. For custom extractors, verify the required return types and confirm the algorithm is appropriate for the input images.

### Inconsistent angular sample counts

If downsampling is disabled, successful detections must use consistent analysis-contour lengths. Enable `--downsample` or modify the extractor to return consistent sampling.

### A QC configuration rejects every detection

This is a QC outcome, not an extraction failure. The `.npz` checkpoint remains valid and can be evaluated again with different QC settings.

---

## Citation

If VesEdge contributes to a publication, please cite the associated manuscript and software repository.
