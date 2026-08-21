# VesEdge CLI

VesEdge is a command-line tool for extracting vesicle contours from ND2 microscopy videos. It reads one or more `.nd2` files, runs an edge extractor on each frame, performs frame- and trajectory-level quality control, optionally writes a visual QC GIF, and saves accepted contours to a NumPy `.npy` file.

## Features

- Process a single ND2 file or every ND2 file in a directory
- Optional recursive directory traversal
- Default VesEdge edge extraction
- User-supplied edge extraction functions
- Optional angular downsampling
- Frame-level curvature quality control
- Trajectory-level center/radius population quality control
- Rerunnable quality control through the Python API without repeating edge extraction
- Versioned `.npz` checkpoints for later QC reanalysis
- GIF output for visual inspection
- NumPy output for downstream analysis

---

## Installation

Install VesEdge into your Python environment.

```bash
git clone <repository-url>
cd VesMod
pip install .
```

Verify that the CLI is available:

```bash
vesedge --help
```

---

## Quick Start

Process one ND2 file with a microscope calibration of 13.44 pixels per micron, downsampling to 180 evenly spaced angular bins, and using the default quality-control settings:

```bash
vesedge sample.nd2 --pixels-per-micron 13.44 --downsample --n-samples 180
```

---

## Command Line Interface Arguments

### File Selection Options

#### Single file

```bash
vesedge sample.nd2
```

#### Directory

```bash
vesedge ./videos
```

When `INPUT_PATH` is a directory, VesEdge processes files matching:

```text
*.nd2
```

in that directory.

#### Recursive directory search

```bash
vesedge ./videos --recursive
```

With `--recursive`, VesEdge processes files matching:

```text
**/*.nd2
```

---

### Microscope Calibration

#### `--pixels-per-micron`

```bash
vesedge sample.nd2 --pixels-per-micron 13.44
```

This value is the microscope calibration in:

```text
pixels per micron
```

For example, if 1 micron corresponds to 13.44 pixels in the image, use `13.44`.

The saved `.npy` file stores radial distances in microns.

Default:

```text
1
```

---

### Downsampling

#### `--downsample`

Using a fixed number of angular samples is recommended when comparing contours across datasets or extraction algorithms. When `--downsample` is used, VesEdge resamples each contour to a fixed number of evenly spaced angular samples.

```bash
vesedge sample.nd2 --downsample
```

Downsampling occurs before quality control. Consequently, using `--downsample` may affect which frames are classified as reliable or unreliable.

Default:

```text
False
```

#### `--n-samples`

The default is to downsample to 120 samples. To specify a different number, use `--n-samples`:

```bash
vesedge sample.nd2 --downsample --n-samples 360
```

Default:

```text
120
```

---

### Frame-Level Curvature Quality Control

#### `--curvature-threshold`

```bash
vesedge sample.nd2 --curvature-threshold 5
```

After edge extraction and optional downsampling, VesEdge computes a wrapped finite second difference of each contour. If the absolute finite second difference exceeds the curvature threshold, the detection fails curvature QC.

Larger values are more permissive. Smaller values are more stringent.

Default:

```text
5
```

---

### Trajectory-Level Population Quality Control

After frame-level QC, VesEdge compares the centers and median radii of eligible detections across the video. This check is intended to identify a small, distinct population of detections, such as frames in which the edge extractor switched from the vesicle to another object.

Population QC uses three features for each eligible detection:

- x-coordinate of the detected vesicle center
- y-coordinate of the detected vesicle center
- median vesicle radius

The features are robustly scaled and compared using one- and two-component Gaussian mixture models.

#### `--population-bic-threshold`

Minimum improvement in Bayesian information criterion (BIC) required before VesEdge prefers a two-population model:

```bash
vesedge sample.nd2 --population-bic-threshold 10
```

Larger values require stronger evidence for two populations.

Default:

```text
10
```

#### `--max-minor-population-fraction`

Maximum fraction of eligible detections that may belong to the smaller population for that population to be rejected:

```bash
vesedge sample.nd2 --max-minor-population-fraction 0.25
```

The value must be greater than or equal to 0 and less than 0.5.

Default:

```text
0.25
```

#### `--no-population-qc`

Disable trajectory-level population QC:

```bash
vesedge sample.nd2 --no-population-qc
```

Population QC is enabled by default.

---

## Rerunning Quality Control in Python

`VesicleVideo.extract_edges()` stores successful and failed extraction results and then runs the configured QC checks. QC can subsequently be rerun on the existing detections without repeating edge extraction.

```python
from vesmod.VesEdge import EdgeQCConfig

new_qc_config = EdgeQCConfig(
    curvature_threshold=8.0,
    population_bic_threshold=10.0,
    max_minor_population_fraction=0.25,
)

video.run_qc(new_qc_config)
```

When `run_qc()` is called, previously stored QC flags, curvature scores, population labels, population probabilities, and trajectory-level population results are cleared before the enabled checks are rerun. Passing a new `EdgeQCConfig` also replaces `video.qc_config`. If no config is supplied, the current `video.qc_config` is reused.

### Saving a checkpoint for later reanalysis

The regular `.npy` output contains only detections that passed the currently enabled QC checks. It is therefore a filtered analysis product and cannot restore detections that were previously rejected.

If you may want to change QC settings later, save a checkpoint while the `VesicleVideo` is still available:

```python
video.save_checkpoint("sample_checkpoint.npz")
```

The versioned `.npz` checkpoint preserves:

- successful detections, including detections rejected by the current QC settings
- extraction failures and their original frame ordering
- detected vesicle centers
- native extracted contours
- analysis contours
- physical radii in microns
- extraction and QC configuration values

Raw image frames and existing QC results are not stored. QC results are recomputed when the checkpoint is loaded.

Reload the checkpoint with the stored QC settings:

```python
video = VesicleVideo.from_checkpoint(
    "sample_checkpoint.npz"
)
```

or supply new QC settings immediately:

```python
video = VesicleVideo.from_checkpoint(
    "sample_checkpoint.npz",
    qc_config=new_qc_config,
)
```

A checkpoint-loaded `VesicleVideo` has `frames=None`. It can rerun QC and save new accepted `.npy` outputs, but it cannot rerun edge extraction or generate an image GIF because the source image frames are not present.

For example:

```python
video.save_edge_to_npy(
    "sample_reanalyzed.npy"
)
```

Checkpoint save/load is currently part of the Python API. The VesEdge CLI still performs extraction and QC together for each ND2 input and does not yet create or consume checkpoint files.

---

### Output Options

#### `--no-gif`

By default, VesEdge writes a GIF for each processed ND2 file.

Disable GIF output:

```bash
vesedge sample.nd2 --no-gif
```

The GIF is intended for visual quality control. Successful edge detections can be displayed even when they fail QC so that rejected contours can be inspected.

Default:

```text
False
```

#### `--overwrite`

By default, VesEdge skips an input file if any expected output file already exists. The expected outputs are the `.npy` file and, unless `--no-gif` is used, the `.gif` file.

Force reprocessing:

```bash
vesedge sample.nd2 --overwrite
```

Default:

```text
False
```

---

### Edge Extraction Algorithm

By default, VesEdge uses the built-in edge extraction algorithm exposed as `vesmod.VesEdge:extract_edge_from_frame`. The user may instead supply a custom edge extraction algorithm. See `custom_edge_extraction_algorithms.README.md` for details.

#### `--extractor`

Load an extractor using `module:function` syntax:

```bash
vesedge sample.nd2 --extractor my_module:my_extractor
```

#### `--extractor-file` and `--extractor-name`

Load an extractor directly from a Python file:

```bash
vesedge sample.nd2 --extractor-file ./my_extractor.py --extractor-name extract_edge_from_frame
```

The default function name is `extract_edge_from_frame`.

---

## Output Files

For an input file:

```text
sample.nd2
```

VesEdge writes:

```text
sample.gif
sample.npy
```

unless GIF output is disabled with `--no-gif`.

### GIF File

```text
sample.gif
```

The GIF shows successfully detected contours overlaid on the image frames. A contour that was extracted successfully can still appear in the GIF even if it was rejected by QC. Frames for which edge extraction itself failed have no contour to display.

### NumPy File

```text
sample.npy
```

Load the output with:

```python
import numpy as np

edges = np.load("sample.npy")
```

The array has shape:

```python
(n_accepted_frames, n_theta)
```

where:

- `n_accepted_frames` is the number of successful detections that passed all enabled quality-control checks
- `n_theta` is the number of angular samples in each saved contour

Distances are stored in microns.

If `--downsample` is enabled, the saved contours are the downsampled contours. These are also the contours used for quality control.

Only accepted frames are written to the `.npy` file.

---

## Frame Results

Each input frame produces either a successful edge detection or an edge extraction failure.

Successful detections are then evaluated by the enabled quality-control checks. A successful detection may therefore still be rejected from the saved `.npy` output.

The GIF can retain successfully extracted contours even when they fail QC, allowing rejected detections to be inspected visually. Frames for which edge extraction itself failed have no contour to display.

Only successful detections that pass all enabled QC checks are written to the `.npy` file.

---

## Troubleshooting

### `No .nd2 files found`

Check that:

- the input path exists
- the files have the `.nd2` extension
- `--recursive` is used if the files are in subdirectories

### `Expected an .nd2 file`

This occurs when `INPUT_PATH` is a file, but its suffix is not `.nd2`.

### Custom extractor cannot be imported

For `--extractor-file`, check that:

- the file path exists
- the file contains the function named by `--extractor-name`
- the function is callable

For `--extractor`, check that:

- the string has the form `module:function`
- the module is importable from the current Python environment
- the named function exists
- the named object is callable

### VesEdge reports no successful detections

This means edge extraction failed on every frame. For a custom extractor, check that it satisfies the required extractor interface and that it is appropriate for the input images.

### VesEdge reports no frames passed quality control

Edge extraction produced one or more contours, but every successful detection was rejected by the enabled QC checks. Inspect the extraction and QC settings rather than treating this as an extractor-interface failure.

### Extracted edges have inconsistent numbers of angular samples

Without downsampling, all successful extractor outputs must contain the same number of angular samples. Enable `--downsample` or modify a custom extractor so that successful results use a consistent angular sampling.

### Frames are missing from the `.npy` file

Only accepted frames are saved.

Frames may be omitted because:

- edge extraction raised an exception
- the returned contour was invalid
- the finite second difference exceeded `--curvature-threshold`
- trajectory-level population QC classified the detection as part of a small outlier population

Inspect the GIF output to determine which successfully extracted contours were rejected.

### Changing `--n-samples` changes accepted frames

This can happen because downsampling occurs before quality control. Changing the number of angular samples changes the contour used for the finite-second-difference check and can therefore change frame-level QC results.

---

## Citation

If VesEdge contributes to a publication, please cite the associated manuscript and software repository.
