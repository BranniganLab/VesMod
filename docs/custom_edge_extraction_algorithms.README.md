# Custom Edge Extractors

VesEdge can run a user-provided edge extraction function instead of the default extractor.

Custom extractors are useful for:

- testing new algorithms
- comparing edge detection methods
- adapting VesEdge to new imaging conditions
- benchmarking against the default VesEdge extractor

Custom extractors can be provided either as an importable Python function or as a standalone Python file.

---

## Required Custom Extractor Interface

A custom edge extractor must have this interface:

```python
def my_edge_extractor(frame):
    ...
    return r_vals, vesicle_center
```

The function receives one two-dimensional NumPy image frame at a time.

### `r_vals`

`r_vals` must be a one-dimensional NumPy `ndarray` of radial distances from the vesicle center to the detected edge.

Requirements:

- NumPy `ndarray`;
- one-dimensional;
- numeric;
- evenly spaced in angle;
- ordered from `0` to `2π` exclusive;
- measured in pixels.

Example:

```python
import numpy as np

r_vals = np.full(360, 100.0)
```

### `vesicle_center`

Return the center in NumPy image-coordinate order:

```python
vesicle_center = (row_coordinate, column_coordinate)
```

or equivalently:

```python
vesicle_center = (y_center, x_center)
```

Example:

```python
vesicle_center = (245.3, 261.8)
```

The function must return exactly:

```python
return r_vals, vesicle_center
```

VesEdge converts this returned `(row, column)` pair to the internal Cartesian `(x, y)` contour origin. Custom extractors should not perform that swap themselves.

---

## Minimal Example

Create `constant_radius_extractor.py`:

```python
import numpy as np


def constant_radius_extractor(frame):
    """Return a constant-radius contour for CLI testing."""
    center_y = frame.shape[0] / 2
    center_x = frame.shape[1] / 2
    vesicle_center = (center_y, center_x)

    radius_pixels = min(frame.shape) / 4
    r_vals = np.full(360, radius_pixels)

    return r_vals, vesicle_center
```

Run it during the extraction stage:

```bash
vesedge extract "sample.nd2" \
    --extractor-file ./constant_radius_extractor.py \
    --extractor-name constant_radius_extractor
```

This writes a QC-independent `.npz` checkpoint. Apply QC later with `vesedge qc`.

---

## Importable Module

Use `--extractor` with `module:function` syntax:

```bash
vesedge extract "sample.nd2" \
    --extractor my_package.my_module:my_edge_extractor
```

Default:

```text
vesmod.VesEdge:extract_edge_from_frame
```

If `--extractor-file` is supplied, the function from that file takes precedence over `--extractor`.

---

## Extraction Failure Behavior

A good extractor should:

- return a contour whenever the vesicle edge can be detected;
- raise an exception when extraction genuinely fails;
- return radii in pixels;
- return the center in `(row, column)` order;
- use consistent angular sampling across successful frames when downsampling is disabled.

VesEdge catches exceptions raised on individual frames, records those frames as `EdgeDetectionFailure`, and continues processing later frames. It does not print per-frame tracebacks from the library layer.

After all frames are processed, extraction fails if:

- no frame produced a successful detection; or
- successful analysis contours have inconsistent angular sample counts.

When every frame fails, the raised error includes the recorded extractor error messages so a custom-extractor failure can be diagnosed without relying on printed tracebacks.

Quality-control outcomes are **not** extraction failures. `VesicleVideo.extract_edges()` and `vesedge extract` do not run QC. They produce reusable extraction results that can later be evaluated under different QC configurations.

For example:

```bash
vesedge extract "sample.nd2" \
    --extractor-file ./my_extractor.py \
    --extractor-name my_edge_extractor \
    --output-dir ./checkpoints

vesedge qc "./checkpoints" \
    --curvature-threshold 10 \
    --output-dir ./results/qc_standard
```

This separation is useful when developing an extractor because changes to the extraction algorithm can be distinguished from changes to the curvature-QC policy.

---

## Python API

```python
from vesmod.VesEdge import (
    EdgeExtractionConfig,
    VesicleVideo,
)

video = VesicleVideo(frames)
edges = video.extract_edges(
    my_edge_extractor,
    EdgeExtractionConfig(
        pixels_per_micron=13.44,
        n_angular_samples=120,
    ),
)
```

Save the extraction state independently of QC:

```python
edges.save_checkpoint("sample.npz")
```

Later, reload it and run curvature QC without invoking the extractor again:

```python
from vesmod.VesEdge import EdgeQCConfig, VesicleEdges

edges = VesicleEdges.from_checkpoint("sample.npz")
edges.run_qc(
    EdgeQCConfig(
        curvature_threshold=10.0,
    )
)
edges.save_edge_to_npy("sample.npy")
```

The checkpoint stores pixel-space contours plus the extraction calibration; physical radii are derived when accepted contours are exported. This separation allows extractor development and QC tuning to be evaluated independently.
