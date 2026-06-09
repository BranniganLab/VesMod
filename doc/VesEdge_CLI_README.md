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

Process one ND2 file:

```bash
vesedge sample.nd2
```

Process every ND2 file in a directory:

```bash
vesedge ./videos
```

Process every ND2 file in a directory and its subdirectories:

```bash
vesedge ./videos --recursive 
```

---

## Command Line Interface

```bash
vesedge INPUT_PATH [OPTIONS]
```

`INPUT_PATH` must be either:

- a single `.nd2` file
- a directory containing `.nd2` files

---


## File Selection Options

### Single file

```bash
vesedge sample.nd2
```

### Directory

```bash
vesedge ./videos
```

When `INPUT_PATH` is a directory, VesEdge processes files matching:

```text
*.nd2
```

in that directory.

### Recursive directory search

```bash
vesedge ./videos --recursive
```

With `--recursive`, VesEdge processes files matching:

```text
**/*.nd2
```

---


## Downsampling

### `--downsample`

```bash
vesedge sample.nd2 --downsample
```

When `--downsample` is used, VesEdge resamples each contour to a fixed number of evenly spaced angular samples.

### `--n_samples`

```bash
vesedge sample.nd2 --downsample --n_samples 360
```

Default:

```text
120
```

Downsampling occurs before curvature-based quality control. Consequently, changing `--n_samples` may affect which frames are classified as reliable or unreliable.

Using a fixed value of `--n_samples` is recommended when comparing contours across datasets or extraction algorithms.


---


## Curvature-Based Quality Control

### `--curvature-threshold`

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


### `--micron-to-pixel-ratio`

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

## GIF Output

By default, VesEdge writes a GIF for each processed ND2 file.

Disable GIF output:

```bash
vesedge sample.nd2 --no-gif
```

The GIF is intended for visual quality control. Accepted and unreliable traces are displayed differently so that rejected frames can be inspected.

---

## Overwriting Existing Output

### `--overwrite`

By default, VesEdge skips an input file if the corresponding GIF already exists.

Force reprocessing:

```bash
vesedge sample.nd2 --overwrite
```

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

## Custom Edge Extractors

VesEdge can run a user-provided edge extraction function instead of the default extractor.

Custom extractors are useful for:

- testing new algorithms
- comparing edge detection methods
- adapting VesEdge to new imaging conditions
- benchmarking against the default VesEdge extractor

Custom extractors can be provided in two ways:

1. as an importable Python module and function
2. as a standalone Python file containing a function

---

## Required Custom Extractor Interface

A custom edge extractor must be a Python function with this interface:

```python
def my_edge_extractor(frame):
    ...
    return r_vals, vesicle_center
```

The function receives one image frame at a time.

### Input: `frame`

`frame` is a two-dimensional NumPy array containing pixel intensities from one microscopy image.

Example:

```python
def my_edge_extractor(frame):
    print(frame.shape)
```

A typical frame shape might be:

```text
(512, 512)
```

### Output: `r_vals`

`r_vals` must be a one-dimensional list or NumPy array of radial distances from the vesicle center to the vesicle edge.

Requirements:

- one-dimensional
- numeric
- evenly spaced in angle
- ordered from `0` to `2π`
- measured in pixels, not microns

Example:

```python
import numpy as np

r_vals = np.full(360, 100.0)
```

### Output: `vesicle_center`

`vesicle_center` is the center used to define the polar contour.

Use NumPy image-coordinate order:

```python
vesicle_center = (row_coordinate, column_coordinate)
```

equivalently:

```python
vesicle_center = (y_center, x_center)
```

Example:

```python
vesicle_center = (245.3, 261.8)
```

The extractor must return exactly:

```python
return r_vals, vesicle_center
```

---

## Minimal Custom Extractor Example

This example returns a constant-radius circle. It is useful for testing that the CLI can load a custom extractor, but it is not a real edge detection algorithm.

Create a file named `constant_radius_extractor.py`:

```python
import numpy as np


def constant_radius_extractor(frame):
    """Return a constant-radius circular contour for CLI testing."""
    center_y = frame.shape[0] / 2
    center_x = frame.shape[1] / 2
    vesicle_center = (center_y, center_x)

    n_theta = 360
    radius_pixels = min(frame.shape) / 4
    r_vals = np.full(n_theta, radius_pixels)

    return r_vals, vesicle_center
```

Run:

```bash
vesedge sample.nd2 \
    --extractor-file ./constant_radius_extractor.py \
    --extractor-name constant_radius_extractor
```

---

## Using an Extractor From a Python File

Use `--extractor-file` when your extractor is in a standalone Python file.

```bash
vesedge sample.nd2 \
    --extractor-file ./my_extractor.py \
    --extractor-name my_edge_extractor
```

The file must contain the function named by `--extractor-name`.

Example `my_extractor.py`:

```python
import numpy as np


def my_edge_extractor(frame):
    """Extract an edge from one frame."""
    center_y = frame.shape[0] / 2
    center_x = frame.shape[1] / 2
    vesicle_center = (center_y, center_x)

    r_vals = np.full(360, min(frame.shape) / 4)

    return r_vals, vesicle_center
```

If `--extractor-file` is provided, VesEdge uses the function from that file. In the current CLI behavior, this takes precedence over `--extractor`.

---

## Using an Extractor From an Importable Module

Use `--extractor` when your extractor is in an importable Python module.

The syntax is:

```text
module:function
```

Example:

```bash
vesedge sample.nd2 \
    --extractor my_package.my_module:my_edge_extractor
```

If you have a file named `my_extractors.py` on your Python path:

```python
import numpy as np


def my_edge_extractor(frame):
    center_y = frame.shape[0] / 2
    center_x = frame.shape[1] / 2
    vesicle_center = (center_y, center_x)

    r_vals = np.full(360, min(frame.shape) / 4)

    return r_vals, vesicle_center
```

you can run:

```bash
vesedge sample.nd2 \
    --extractor my_extractors:my_edge_extractor
```

Default:

```text
vesmod.VesEdge:extract_edge_from_frame
```

---

## Practical Extractor Recommendations

A good custom extractor should:

- return a contour for every frame where the vesicle edge can be detected
- raise an exception when extraction genuinely fails
- return `r_vals` in pixels
- return `vesicle_center` in `(row, column)` order
- use a consistent number of angular samples when possible
- avoid returning NaNs unless the frame should be treated as failed or unreliable

VesEdge catches exceptions raised by the extractor on individual frames, marks those frames as failed, and continues processing subsequent frames.

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
