# Custom Edge Extractors

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

`r_vals` must be a one-dimensional NumPy `ndarray` of radial distances
from the vesicle center to the vesicle edge.

Requirements:

- a NumPy `ndarray`
- one-dimensional
- numeric
- evenly spaced in angle
- ordered from `0` to `2π` (exclusive)
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

