# VesEdge CLI

The VesEdge CLI extracts vesicle contours from ND2 microscopy videos, applies the configured quality-control checks, and writes QC-filtered `.npy` files for EdgeMod. The underlying Python API now separates raw video data, extracted-edge state, and QC so extraction results can also be checkpointed and re-evaluated without rerunning image processing.

## Quick Start

```bash
vesedge sample.nd2 --pixels-per-micron 13.44 --downsample --n-samples 120
```

By default this writes:

```text
sample.npy
sample.gif
```

The `.npy` file contains only contours accepted by the configured QC checks, with radii in microns. The GIF overlays extracted contours on the original frames.

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

Specifies the microscope calibration used when accepted analysis contours are converted from pixels to microns for `.npy` output.

Default: `1.0`.

---

## Angular Sampling

Use `--downsample` to resample successful contours to a fixed angular grid before QC and downstream analysis:

```bash
vesedge sample.nd2 --downsample --n-samples 120
```

`--n-samples` defaults to `120`. Without `--downsample`, the native angular sampling returned by the extractor is retained, and successful detections must have consistent analysis-contour lengths.

---

## Quality Control

### Curvature QC

```bash
vesedge sample.nd2 --curvature-threshold 5
```

`--curvature-threshold` sets the maximum allowed wrapped finite second difference of an analysis contour. Default: `5`.

### Population QC

```bash
vesedge sample.nd2 \
    --population-bic-threshold 10 \
    --max-minor-population-fraction 0.25
```

Population QC compares frame centers and median radii across the trajectory and can reject a sufficiently small, distinct population. Disable it with:

```bash
vesedge sample.nd2 --no-population-qc
```

---

## Extraction Algorithm

By default VesEdge uses:

```text
vesmod.VesEdge:extract_edge_from_frame
```

Use an importable custom extractor with:

```bash
vesedge sample.nd2 --extractor my_package.my_module:my_edge_extractor
```

or a function from a Python file with:

```bash
vesedge sample.nd2 \
    --extractor-file ./my_extractor.py \
    --extractor-name my_edge_extractor
```

See `custom_edge_extraction_algorithms.README.md` for the required interface.

---

## GIF and Existing Outputs

Disable GIF output with:

```bash
vesedge sample.nd2 --no-gif
```

By default an input is skipped if an expected `.npy` or GIF output already exists. Force reprocessing with:

```bash
vesedge sample.nd2 --overwrite
```

---

## Python API: Separate Extraction and QC

The Python data model separates raw frames from reusable extraction results:

```python
from vesmod.VesEdge import (
    EdgeExtractionConfig,
    EdgeQCConfig,
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

`VesicleVideo.extract_edges()` performs extraction only and returns a `VesicleEdges` object. QC is explicit:

```python
qc_config = EdgeQCConfig(
    curvature_threshold=10.0,
    population_bic_threshold=10.0,
    max_minor_population_fraction=0.25,
)

edges.run_qc(qc_config)
edges.save_edge_to_npy("sample.npy")
```

A completed run is summarized by `edges.qc_result`, a `VesicleQCResult` containing the configuration and the results of each enabled QC stage. Curvature results are available through `edges.qc_result.curvature`, and population-analysis results through `edges.qc_result.population`. Individual detections retain their own detailed QC annotations in `EdgeDetection.qc`.

Calling `run_qc()` again clears the prior derived QC state and aggregate result before applying the new configuration.

---

## Checkpointing Extraction Results

The Python API can save QC-independent extraction state:

```python
edges.save_checkpoint("sample.npz")
```

and restore it later:

```python
from vesmod.VesEdge import VesicleEdges

edges = VesicleEdges.from_checkpoint("sample.npz")
```

A checkpoint stores the extraction configuration, successful native and analysis contours, extraction failures, and frame ordering. QC settings and QC results are intentionally not stored. Physical radii are derived from the stored pixel-space analysis contours and `pixels_per_micron` calibration.

This makes it possible to evaluate several QC configurations without rerunning edge extraction:

```python
edges.run_qc(qc_config_a)
edges.save_edge_to_npy("qc_a/sample.npy")

edges.run_qc(qc_config_b)
edges.save_edge_to_npy("qc_b/sample.npy")
```

A future CLI update will expose this checkpoint/re-QC workflow directly from the command line. In this PR, the existing CLI continues to perform extraction, QC, and `.npy` export in one invocation.

---

## Troubleshooting

### No successful detections

If extraction fails on every frame, VesEdge reports the per-frame extractor errors and does not continue to QC.

### Inconsistent angular sample counts

If downsampling is disabled, successful detections must use consistent analysis-contour lengths. Enable `--downsample` or modify the extractor to return consistent sampling.

### No frames pass QC

This is distinct from extraction failure: the extractor produced usable contours, but the selected QC configuration rejected all successful detections.

---

## Citation

If VesEdge contributes to a publication, please cite the associated manuscript and software repository.
