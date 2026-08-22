# VesEdge CLI

The VesEdge CLI separates image-dependent edge extraction from quality control.

The intended workflow is:

```text
.nd2 microscopy videos
        │
        │ vesedge extract
        ▼
QC-independent .npz checkpoints
        │
        ├── vesedge qc with configuration A ──▶ filtered .npy files
        ├── vesedge qc with configuration B ──▶ filtered .npy files
        └── vesedge qc with configuration C ──▶ filtered .npy files
                                                     │
                                                     ▼
                                                  EdgeMod
```

This allows expensive image processing to be run once while QC settings are evaluated repeatedly and reproducibly.

## Quick Start

Extract a directory of ND2 videos:

```bash
vesedge extract ./videos \
    --pixels-per-micron 13.44 \
    --downsample \
    --n-samples 120 \
    --output-dir ./checkpoints
```

Apply one QC configuration:

```bash
vesedge qc ./checkpoints \
    --curvature-threshold 5 \
    --population-bic-threshold 10 \
    --max-minor-population-fraction 0.25 \
    --output-dir ./results/qc_standard
```

Analyze the filtered results:

```bash
edgemod ./results/qc_standard
```

---

# `vesedge extract`

`vesedge extract` reads ND2 microscopy videos and writes reusable VesEdge `.npz` checkpoints. It does **not** run quality control.

## Input Selection

Extract one video:

```bash
vesedge extract sample.nd2
```

Extract all ND2 files in a directory:

```bash
vesedge extract ./videos
```

Search recursively:

```bash
vesedge extract ./videos --recursive
```

Directory discovery treats file suffixes case-insensitively, so `.nd2` and `.ND2` inputs are handled consistently.

## Output Location

By default, `.npz` and GIF outputs are written beside each ND2 input.

Use a dedicated checkpoint directory with:

```bash
vesedge extract ./videos --output-dir ./checkpoints
```

For a file named `sample.nd2`, extraction normally produces:

```text
sample.npz
sample.gif
```

When recursive input discovery is used with `--output-dir`, VesEdge preserves each input file's path relative to the selected input directory. For example, `videos/a/sample.nd2` and `videos/b/sample.nd2` produce `checkpoints/a/sample.npz` and `checkpoints/b/sample.npz` rather than colliding.

The `.npz` file is the reusable extraction product. It contains successful detections, extraction failures, frame ordering, pixel-space native and analysis contours, and the extraction configuration. It contains no QC decisions.

The GIF overlays the extracted full contour on the original image frames for visual inspection.

Disable GIF creation with:

```bash
vesedge extract sample.nd2 --no-gif
```

Existing checkpoints are not overwritten unless:

```bash
vesedge extract sample.nd2 --overwrite
```

## Microscope Calibration

```bash
vesedge extract sample.nd2 --pixels-per-micron 13.44
```

`--pixels-per-micron` stores the microscope calibration in the checkpoint. Accepted contours are converted from pixels to microns later, when QC output is exported to `.npy`.

Default: `1.0`.

## Angular Sampling

Use fixed angular sampling with:

```bash
vesedge extract sample.nd2 --downsample --n-samples 120
```

`--n-samples` defaults to `120` and is used only when `--downsample` is provided.

Without `--downsample`, the extractor's native angular sampling is retained as the analysis contour. Successful detections must therefore have consistent analysis-contour lengths.

## Extraction Algorithm

The default extractor is:

```text
vesmod.VesEdge:extract_edge_from_frame
```

Use an importable custom extractor with:

```bash
vesedge extract sample.nd2 \
    --extractor my_package.my_module:my_edge_extractor
```

or a Python source file with:

```bash
vesedge extract sample.nd2 \
    --extractor-file ./my_extractor.py \
    --extractor-name my_edge_extractor
```

See `custom_edge_extraction_algorithms.README.md` for the required extractor interface.

---

# `vesedge qc`

`vesedge qc` loads one or more VesEdge `.npz` checkpoints, applies a single QC configuration, and writes the accepted contours as `.npy` files for EdgeMod.

The command requires an output directory:

```bash
vesedge qc ./checkpoints --output-dir ./results/qc_standard
```

Use a **different output directory for each QC configuration**. This keeps the `.npy` data, exact QC settings, resolved checkpoint selection, and QC summary together as one reproducible analysis condition.

## Input Selection

QC one checkpoint:

```bash
vesedge qc sample.npz --output-dir ./results/qc_standard
```

QC all checkpoints in a directory:

```bash
vesedge qc ./checkpoints --output-dir ./results/qc_standard
```

Search recursively with:

```bash
vesedge qc ./checkpoints --recursive --output-dir ./results/qc_standard
```

Directory discovery treats `.npz` suffixes case-insensitively. For recursive directory inputs, output `.npy` files preserve the checkpoint paths relative to the selected checkpoint directory, preventing equal filenames in different subdirectories from colliding.

## Curvature QC

```bash
vesedge qc ./checkpoints \
    --curvature-threshold 5 \
    --output-dir ./results/qc_standard
```

The curvature threshold is the maximum allowed wrapped finite second difference of an analysis contour.

Default: `5.0`.

Disable curvature QC with:

```bash
vesedge qc ./checkpoints \
    --no-curvature-qc \
    --output-dir ./results/no_curvature
```

## Population QC

```bash
vesedge qc ./checkpoints \
    --population-bic-threshold 10 \
    --max-minor-population-fraction 0.25 \
    --output-dir ./results/qc_standard
```

Population QC compares each successful detection's vesicle center and median analysis radius across the trajectory. A two-component Gaussian-mixture model is used only when enough usable detections are available.

Disable population QC with:

```bash
vesedge qc ./checkpoints \
    --no-population-qc \
    --output-dir ./results/no_population
```

## QC Outputs

For checkpoints `sample01.npz` and `sample02.npz`, the command:

```bash
vesedge qc ./checkpoints --output-dir ./results/qc_standard
```

creates:

```text
results/qc_standard/
├── sample01.npy
├── sample02.npy
├── vesedge_qc.json
└── qc_summary.csv
```

### Filtered `.npy` files

Each `.npy` contains only contours accepted under the current QC configuration, with radial distances converted to microns. These files are directly consumable by EdgeMod.

If a checkpoint completes QC but no frames are accepted, no `.npy` is written for that checkpoint. The outcome is still represented in `qc_summary.csv`.

### `vesedge_qc.json`

This file records:

- the resolved checkpoint input path;
- whether recursive discovery was enabled;
- the resolved manifest of checkpoints selected for the batch;
- `curvature_threshold`;
- `population_bic_threshold`;
- `max_minor_population_fraction`;
- whether curvature QC was enabled;
- whether population QC was enabled.

Consequently, recursive and non-recursive runs, or runs resolving to different checkpoint sets, have different provenance even if their QC thresholds are identical.

If an output directory already contains incompatible provenance, VesEdge refuses to mix the results unless `--overwrite` is explicitly supplied. An incompatible overwrite first removes VesEdge-managed `.npy` outputs and QC metadata from the previous batch so orphaned filtered results cannot remain under the new provenance.

### `qc_summary.csv`

The summary contains one row per selected checkpoint with:

- total source frames;
- successful edge detections;
- extraction failures;
- curvature rejections;
- population rejections;
- accepted frames;
- accepted fraction;
- processing status;
- any load or QC error message.

A checkpoint that cannot be loaded receives a `load_error` row with zero counts and the loading error. Therefore `qc_summary.csv` is still written when every selected checkpoint fails to load.

This file is intended to make it easy to compare how aggressive different QC configurations are before comparing the downstream EdgeMod results.

## Comparing QC Configurations

A typical sensitivity analysis might use:

```bash
vesedge qc ./checkpoints \
    --curvature-threshold 5 \
    --output-dir ./results/qc_strict

vesedge qc ./checkpoints \
    --curvature-threshold 10 \
    --output-dir ./results/qc_standard

vesedge qc ./checkpoints \
    --curvature-threshold 15 \
    --output-dir ./results/qc_permissive
```

Then run:

```bash
edgemod ./results/qc_strict
edgemod ./results/qc_standard
edgemod ./results/qc_permissive
```

The resulting EdgeMod estimates can be reported alongside each directory's `vesedge_qc.json` and `qc_summary.csv` to show whether the scientific conclusion depends strongly on QC choices.

---

## Python API

The CLI mirrors the Python API separation:

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

Later:

```python
edges = VesicleEdges.from_checkpoint("sample.npz")
edges.run_qc(
    EdgeQCConfig(
        curvature_threshold=10.0,
        population_bic_threshold=10.0,
        max_minor_population_fraction=0.25,
    )
)
edges.save_edge_to_npy("sample.npy")
```

A completed run is summarized by `edges.qc_result`; individual detections retain detailed QC annotations through `EdgeDetection.qc`.

---

## Troubleshooting

### No successful detections during extraction

If extraction fails on every frame, VesEdge reports the extractor errors and does not produce a checkpoint.

### Inconsistent angular sample counts

If downsampling is disabled, successful detections must have consistent analysis-contour lengths. Enable `--downsample` or modify the extractor to return consistent sampling.

### No frames pass QC

This is distinct from extraction failure. The checkpoint remains valid, but the selected QC configuration accepted no frames. The event is recorded in `qc_summary.csv` and no `.npy` is written for that checkpoint.

### QC output directory already contains another configuration or input selection

Choose another `--output-dir`. Reusing one directory for incompatible QC provenance is intentionally blocked unless `--overwrite` is used.

---

## Citation

If VesEdge contributes to a publication, please cite the associated manuscript and software repository.