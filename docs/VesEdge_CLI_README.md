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
        ├── vesedge qc with configuration A ──▶ filtered .npy files ──▶ EdgeMod
        ├── vesedge qc with configuration B ──▶ filtered .npy files ──▶ EdgeMod
        └── vesedge internal-structures ───────▶ interior measurements
```

This allows expensive image processing to be run once while QC settings are evaluated repeatedly and reproducibly. Curvature is currently the only built-in frame-rejection rule.

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

The `.npz` file is the reusable extraction product. It contains successful detections, extraction failures, frame ordering, pixel-space native and analysis contours, the extraction configuration, and source-video provenance when available. It contains no QC decisions.

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

See the [custom extractor guide](custom_edge_extraction_algorithms.README.md) for the required interface.

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

The curvature threshold is the maximum allowed absolute wrapped finite second difference of an analysis contour. A successful detection is rejected when its score is greater than the threshold.

Default: `5.0`.

Disable curvature QC with:

```bash
vesedge qc ./checkpoints \
    --no-curvature-qc \
    --output-dir ./results/no_curvature
```

With curvature QC disabled, every successfully extracted detection is exported. Extraction failures remain absent because they do not contain contours.

VesEdge no longer performs GMM-based population QC. The removed options `--population-bic-threshold`, `--max-minor-population-fraction`, and `--no-population-qc` are invalid and produce an argument error. VesEdge does not currently attempt to identify dust-particle detections automatically; inspect extraction GIFs before downstream analysis.

## QC Outputs

For checkpoints `sample01.npz` and `sample02.npz`, the command:

```bash
vesedge qc ./checkpoints --output-dir ./results/qc_standard
```

normally creates:

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
- whether curvature QC was enabled.

Consequently, recursive and non-recursive runs, or runs resolving to different checkpoint sets, have different provenance even if their QC thresholds are identical.

### `qc_summary.csv`

The summary contains one row per selected checkpoint with:

- total source frames;
- successful edge detections;
- extraction failures;
- curvature rejections;
- accepted frames;
- accepted fraction;
- processing status;
- any load or QC error message.

A checkpoint that cannot be loaded receives a `load_error` row with zero counts and the loading error. Therefore `qc_summary.csv` is still written when every selected checkpoint fails to load.

This file is intended to make it easy to compare how aggressive different QC configurations are before comparing the downstream EdgeMod results.

## Existing Outputs and `--overwrite`

If an output `.npy` already exists, VesEdge keeps it unless `--overwrite` is supplied. Reusing an output directory with different provenance is rejected unless `--overwrite` is supplied.

Use a dedicated QC output directory. For an incompatible overwrite, VesEdge recursively removes every `.npy` file under that output directory, along with its QC summary and provenance, before writing the new run. Do not store unrelated NumPy arrays there.

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

# `vesedge internal-structures`

`vesedge internal-structures` is an experimental measurement command independent of curvature QC. It reopens the source ND2 video recorded in each extraction checkpoint and measures bright and dark structures inside each successfully detected full contour. It does not reject edges, alter checkpoints, or assign vesicles to populations.

Run one checkpoint:

```bash
vesedge internal-structures sample.npz \\
    --output-dir ./results/internal_structures
```

Run a recursive checkpoint collection:

```bash
vesedge internal-structures ./checkpoints \\
    --recursive \\
    --output-dir ./results/internal_structures
```

The output directory must be outside the checkpoint directory. This prevents optional mask `.npz` files from being rediscovered as extraction checkpoints.

## Experimental Parameters

```bash
vesedge internal-structures ./checkpoints \\
    --output-dir ./results/internal_structures \\
    --membrane-exclusion-px 5 \\
    --background-sigma-px 8 \\
    --threshold-sigma 4 \\
    --min-region-area-px 9
```

- `--membrane-exclusion-px` erodes the detected interior so the membrane and its optical blur are not counted.
- `--background-sigma-px` controls the spatial scale treated as smooth background. It should be larger than the structures of interest.
- `--threshold-sigma` selects pixels whose absolute local residual exceeds the robust framewise noise estimate, allowing both bright and dark structures.
- `--min-region-area-px` removes connected detections smaller than the requested pixel area.

These defaults are starting values, not calibrated population boundaries. Compare diagnostic overlays across known empty and structured vesicles before interpreting absolute abundance values.

## Source Videos

The checkpoint records the original ND2 path. If videos have moved, place them in one directory and supply:

```bash
vesedge internal-structures ./checkpoints \\
    --video-root /new/video/location \\
    --output-dir ./results/internal_structures
```

The fallback searches recursively and requires exactly one matching filename. For a legacy checkpoint without stored source provenance, VesEdge infers the ND2 filename from the checkpoint stem—for example, `sample.npz` maps to `sample.nd2`. It searches beside the checkpoint and then under `--video-root`; multiple matches are reported as an ambiguity rather than selected silently.

## Outputs

For `sample.npz`, the command writes:

```text
internal_structure_analysis.json
internal_structure_summary.csv
sample_frames.csv
sample_regions.csv
sample_internal_structures.gif
```

Use `--save-masks` to additionally write `sample_masks.npz`. It contains `structure_masks`, a compressed boolean array aligned with the original image coordinates, and `frame_indices`, which identifies the corresponding source frames. Frames without successful measurements are omitted rather than represented as false, structure-free masks.

`sample_frames.csv` reports usable interior area, structured area, structured-area fraction, detected-region count, noise estimate, and status for every source frame. Extraction failures and measurement failures remain explicit rows.

`sample_regions.csv` reports each connected region's bright/dark polarity, area, centroid, bounding box, and signed residual. Centroids and bounding boxes use original image coordinates, not cropped analysis coordinates.

`internal_structure_summary.csv` contains one row per video with the median structured-area fraction, 90th-percentile structured-area fraction, and fraction of analyzed frames containing at least one retained region. These video-level measurements are intended as inputs to later population segmentation; the experimental command does not currently choose a present/absent cutoff.

The diagnostic GIF overlays the detected regions and extracted contour on the original frames. Disable it with `--no-gif`.

As with QC, incompatible provenance is rejected unless `--overwrite` is supplied. An incompatible overwrite removes only files managed by the previous internal-structure batch.

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
    )
)
edges.save_edge_to_npy("sample.npy")
```

A completed run is summarized by `edges.qc_result`; individual detections retain their curvature score and pass/fail flag through `EdgeDetection.qc`.

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

### Population-QC arguments are unrecognized

GMM-based population QC has been removed because a minority radius distribution cannot reliably distinguish bad edge detection from a real change in one vesicle. Remove the population-specific arguments from existing commands.

---

## Citation

If VesEdge contributes to a publication, please cite the associated manuscript and software repository.
