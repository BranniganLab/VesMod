# EdgeMod CLI

EdgeMod is a command-line tool for fitting membrane bending moduli from vesicle contour data. It reads one or more NumPy `.npy` edge files, computes fluctuation spectra, fits membrane mechanical parameters, and writes the results to JSON files for downstream analysis.

## Features

* Process a single `.npy` file or every `.npy` file in a directory
* Optional recursive directory traversal
* Automatic fluctuation spectrum calculation
* Fitting of membrane bending modulus ($k_C$)
* Optional fitting of membrane surface tension ($\sigma$)
* Configurable Fourier fitting range
* Configurable spherical harmonic summation limit
* JSON output for downstream analysis

---

## Installation

Install EdgeMod into your Python environment.

```bash
git clone <repository-url>
cd VesMod
pip install .
```

Verify that the CLI is available:

```bash
edgemod --help
```

---

## Quick Start

Fit membrane mechanical parameters from a single contour file:

```bash
edgemod sample.npy
```

This performs a fit using the default fitting parameters and writes:

```text
sample.json
```

For a VesEdge batch, the recommended input is a QC output directory:

```bash
vesedge qc ./checkpoints --output-dir ./results/qc_standard
edgemod ./results/qc_standard
```

EdgeMod processes the `.npy` files in that directory and ignores the VesEdge provenance files `vesedge_qc.json` and `qc_summary.csv`.

---

## Input Requirements

EdgeMod operates on vesicle contour arrays stored in NumPy `.npy` files.

The recommended input is a contour file produced by `vesedge qc`:

```text
sample.npy
```

Each row represents one accepted video frame and each column represents one angular sample along the vesicle contour.

The expected array shape is:

```python
(n_frames, n_theta)
```

where:

* `n_frames` is the number of accepted contour measurements
* `n_theta` is the number of angular samples per contour

For example:

```python
import numpy as np

edges = np.load("sample.npy")
print(edges.shape)

# (250, 120)
```

In this example, the file contains 250 accepted contours, each sampled at 120 angular positions.

Distances should be stored in microns. VesEdge `.npy` output contains only contours that passed the selected QC configuration and is directly suitable as EdgeMod CLI input.

When comparing QC configurations, keep each set of `.npy` files in its own directory and run EdgeMod separately on each directory. The corresponding `vesedge_qc.json` and `qc_summary.csv` files document which QC settings produced each analysis input.

---

## Command Line Interface Arguments

### File Selection Options

#### Single file

```bash
edgemod sample.npy
```

#### Directory

```bash
edgemod ./edges
```

When `INPUT_PATH` is a directory, EdgeMod processes files matching:

```text
*.npy
```

in that directory.

#### Recursive directory search

```bash
edgemod ./edges --recursive
```

With `--recursive`, EdgeMod processes files matching:

```text
**/*.npy
```

---

### Fourier Mode Fitting Range

#### `--lower-fitting-bound`

```bash
edgemod sample.npy --lower-fitting-bound 3
```

Lowest Fourier mode included in the fit.

Default:

```text
3
```

#### `--upper-fitting-bound`

```bash
edgemod sample.npy --upper-fitting-bound 8
```

First Fourier mode excluded from the fit.

For example:

```text
lower = 3
upper = 8
```

fits modes:

```text
q = 3, 4, 5, 6, 7
```

Default:

```text
8
```

---

### Spherical Harmonic Summation

#### `--lmax`

```bash
edgemod sample.npy --lmax 500
```

Maximum spherical harmonic index used when evaluating the theoretical fluctuation spectrum.

Larger values increase computational cost but may improve convergence in some situations.

Default:

```text
500
```

---

### Surface Tension Treatment

#### Free surface tension fit (default)

By default, EdgeMod fits both:

```text
kC
sigma
```

as free parameters.

#### `--fixed-sigma`

```bash
edgemod sample.npy --fixed-sigma
```

Use a fixed-surface-tension model in which only the bending modulus is optimized.

Default:

```text
False
```

---

### Temperature

#### `--temperature`

```bash
edgemod sample.npy --temperature 295
```

Experimental temperature in Kelvin.

The value is used when converting between thermal energy and membrane mechanical parameters.

Default:

```text
295
```

---

## Output Files

For an input file:

```text
sample.npy
```

EdgeMod writes:

```text
sample.json
```

### JSON Output

The JSON file contains the spectrum metadata and fitted membrane mechanical parameters.

Load the output with:

```python
import json

with open("sample.json") as f:
    results = json.load(f)
```

The exact fields may vary depending on the fitting options used and the version of EdgeMod.

---

## Example Workflows

### Analyze a single vesicle

```bash
edgemod sample.npy
```

### Analyze every vesicle in one VesEdge QC result

```bash
edgemod ./results/qc_standard
```

### Compare two VesEdge QC configurations

```bash
vesedge qc ./checkpoints \
    --curvature-threshold 5 \
    --output-dir ./results/qc_strict

vesedge qc ./checkpoints \
    --curvature-threshold 15 \
    --output-dir ./results/qc_permissive

edgemod ./results/qc_strict
edgemod ./results/qc_permissive
```

### Analyze an entire directory tree

```bash
edgemod ./edges --recursive
```

### Use a different fitting range

```bash
edgemod sample.npy \
    --lower-fitting-bound 4 \
    --upper-fitting-bound 10
```

### Use a different temperature

```bash
edgemod sample.npy --temperature 310
```

### Fit only the bending modulus

```bash
edgemod sample.npy --fixed-sigma
```

---

## Troubleshooting

### `No .npy files found`

Check that:

* the input path exists
* `vesedge qc` produced at least one accepted trajectory
* the files have the `.npy` extension
* `--recursive` is used if the files are located in subdirectories

### `Expected a .npy file`

This occurs when `INPUT_PATH` is a file, but its suffix is not `.npy`.

### Fits produce unexpected values

Potential causes include:

* poor contour quality in the input `.npy` file
* an inappropriate Fourier fitting range
* too few accepted contour measurements
* temperature values inconsistent with the experiment
* sensitivity of the result to the selected VesEdge QC configuration

Inspect the contour data and `qc_summary.csv`, compare alternate QC configurations when appropriate, and consider adjusting:

```text
--lower-fitting-bound
--upper-fitting-bound
--lmax
```

---

## Citation

If EdgeMod contributes to a publication, please cite the associated manuscript and software repository.
