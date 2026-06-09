# VesEdge CLI

VesEdge is a command-line tool for extracting vesicle contours from ND2 microscopy videos. It reads one or more `.nd2` files, runs an edge extractor on each frame, performs contour quality control, optionally writes a visual QC GIF, and saves accepted contours to a NumPy `.npy` file.

## Features

- Process a single ND2 file or every ND2 file in a directory
- Optional recursive directory traversal
- Default VesEdge edge extraction
- User-supplied edge extraction functions
- Optional angular downsampling
- Curvature-based contour quality control
- GIF output for visual inspection
- NumPy output for downstream analysis

---

## Installation

Install VesEdge into your Python environment.


```bash
git clone <repository-url>
cd VesMod/vesmod
pip install .
```

Verify that the CLI is available:

```bash
vesedge --help
```

---

## Quick Start

Process one ND2 file (sample.nd2) with micron : pixel ratio of 1.0 : 13.44, downsampling to 180 evenly-spaced angular bins, and other default values:

```bash
vesedge sample.nd2 --micron-to-pixel-ratio 0.0744 --downsample --n-samples 180
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


### Downsampling

#### `--downsample`

When `--downsample` is used, VesEdge resamples each contour to a fixed number of evenly spaced angular samples.

```bash
vesedge sample.nd2 --downsample
```
Downsampling occurs before curvature-based quality control. Consequently, using `--downsample` may affect which frames are classified as reliable or unreliable.

Using a fixed number of samples is recommended when comparing contours across datasets or extraction algorithms. The default is 120 samples. To specify a different number, use `--n-samples`

#### `--n-samples`

```bash
vesedge sample.nd2 --downsample --n-samples 360
```

---


### Curvature-Based Quality Control

#### `--curvature-threshold`

```bash
vesedge sample.nd2 --curvature-threshold 5
```

After edge extraction and downsampling, VesEdge computes a wrapped finite second difference of each contour. If the absolute finite second difference exceeds the curvature threshold, the frame is classified as unreliable.

Larger values are more permissive.

Smaller values are more stringent.

Default:

```text
5
```

---

### Microscope calibration

#### `--micron-to-pixel-ratio`

```bash
vesedge sample.nd2 --micron-to-pixel-ratio 0.0744
```

This value is the physical calibration of the image in:

```text
microns per pixel
```

For example, if the microscope image has 13.44 pixels per micron, use 0.0744 because:

```text
1 / 13.44 = 0.0744 microns per pixel
```

The saved `.npy` file stores radial distances in microns.


---

### Output options

#### `--no-gif`

By default, VesEdge writes a GIF for each processed ND2 file.

Disable GIF output:

```bash
vesedge sample.nd2 --no-gif
```

The GIF is intended for visual quality control. Accepted and unreliable traces are displayed differently so that rejected frames can be inspected.

---

#### `--overwrite`

By default, VesEdge skips an input file if the corresponding GIF already exists.

Force reprocessing:

```bash
vesedge sample.nd2 --overwrite
```

---

### Edge extraction algorithm

By default, VesEdge will use the built-in edge extraction algorithm contained in 
`vesmod.VesEdge.edge_extractor:extract_edge_from_frame`. The user has the option
of supplying their own edge extraction algorithm. See 
`custom_edge_extraction_algorithms.README.md` for more details.

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

### GIF file

```text
sample.gif
```

The GIF shows the detected contour overlaid on the image frames.

### NumPy file

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

- `n_accepted_frames` is the number of frames that passed quality control
- `n_theta` is the number of angular samples in each saved contour

Distances are stored in microns.

If `--downsample` is enabled, the saved contours are the downsampled contours. These are also the contours used for curvature-based quality control.

Only accepted frames are written to the `.npy` file.

---

## Frame Status Codes

VesEdge assigns an internal status code to each frame.

| Status | Meaning |
| --- | --- |
| `1` | Successful edge extraction |
| `2` | Edge extraction failed |
| `3` | Edge extraction succeeded, but the contour was classified as unreliable |

Only frames with status `1` are saved to the `.npy` output.

Frames with status `3` can still appear in the GIF, which allows visual inspection of contours that were detected but rejected by quality control.

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

### Frames are missing from the `.npy` file

Only accepted frames are saved.

Frames may be omitted because:

- edge extraction raised an exception
- the returned contour contained invalid values
- the finite second difference exceeded `--curvature-threshold`

Inspect the GIF output to determine which frames were rejected.

### Changing `--n_samples` changes accepted frames

This can happen because downsampling occurs before curvature-based quality control. Changing the number of angular samples changes the contour used for the finite-second-difference check.

---

## Citation

If VesEdge contributes to a publication, please cite the associated manuscript and software repository.
