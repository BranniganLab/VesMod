# EdgeMod CLI

EdgeMod fits membrane mechanical parameters from vesicle contour trajectories. It reads one or more NumPy `.npy` edge files, computes fluctuation spectra, selects the Fourier modes used for the physical fit, and writes JSON results for downstream analysis.

EdgeMod supports two q-range strategies:

* **Fixed fitting** uses an explicitly configured lower-inclusive, upper-exclusive q interval.
* **Dynamic fitting** searches within a trusted q interval for the longest contiguous range consistent with q^-3 scaling and rejects the spectrum if no acceptable range is found.

Fixed fitting remains the default CLI behavior.

## Features

* Process a single `.npy` file or every `.npy` file in a directory
* Optional recursive directory traversal
* Automatic fluctuation spectrum calculation
* Fitting of membrane bending modulus ($k_C$)
* Optional fitting of membrane surface tension ($\sigma$)
* Fixed or dynamically selected Fourier fitting ranges
* Explicit dynamic q^-3 acceptance criteria
* Rejection of spectra with no trustworthy q^-3 regime
* Configurable spherical harmonic summation limit
* Spectrum-fit diagnostic PNG, including rejected fit attempts
* JSON output containing fit values, q bounds, configuration, and selection diagnostics
* Separate fixed and dynamic output filenames for direct comparison

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

Fit membrane mechanical parameters from a single contour file using the default fixed q range:

```bash
edgemod sample.npy
```

The default fit uses:

```text
q = 3, 4, 5, 6, 7
lmax = 500
free sigma = True
temperature = 295 K
```

and writes:

```text
sample.json
sample.spectrum_diagnostic.png
```

If fit validation fails, EdgeMod still writes the diagnostic PNG before
raising the validation error. The figure shows the measured and attempted
theoretical spectra, the selected fitting modes, the $q^4$-compensated
spectrum, relative residuals, fitted parameters, and the rejection reason.

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

Each row represents one accepted video frame and each column represents one angular sample along the vesicle contour. The expected shape is:

```python
(n_frames, n_theta)
```

where:

* `n_frames` is the number of accepted contour measurements
* `n_theta` is the number of angular samples per contour

Distances should be stored in microns. VesEdge `.npy` output contains only contours that passed the selected QC configuration and is directly suitable as EdgeMod CLI input.

When comparing VesEdge QC configurations, keep each set of `.npy` files in its own directory and run EdgeMod separately on each directory. The corresponding `vesedge_qc.json` and `qc_summary.csv` files document which QC settings produced each analysis input.

---

## Fixed q-Range Fitting

Fixed fitting is the default.

```bash
edgemod sample.npy \
    --lower-fitting-bound 3 \
    --upper-fitting-bound 8
```

The lower bound is inclusive and the upper bound is exclusive, so the example above fits:

```text
q = 3, 4, 5, 6, 7
```

### `--lower-fitting-bound`

Lowest Fourier mode included in a fixed fit.

Default:

```text
3
```

### `--upper-fitting-bound`

First Fourier mode excluded from a fixed fit.

Default:

```text
8
```

---

## Dynamic q-Range Selection

Dynamic fitting is enabled with:

```bash
--dynamic-range
```

The configured lower and upper fitting bounds become the **trusted search interval**. EdgeMod searches only within that interval; it does not search high-q or very-low-q regions outside the analyst-approved domain.

For example:

```bash
edgemod sample.npy \
    --dynamic-range \
    --lower-fitting-bound 3 \
    --upper-fitting-bound 20 \
    --min-modes 5 \
    --slope-tolerance 0.2 \
    --max-log-rmse 0.1
```

The selector considers contiguous integer-q windows within `3 <= q < 20`. Each candidate window is evaluated in log space using:

1. the fitted log-log power-law slope, which must be sufficiently close to -3; and
2. the RMSE to a fixed q^-3 model, which must be sufficiently small.

Among accepted candidates, the longest range is preferred. If no candidate satisfies the configured criteria, EdgeMod does not run the physical HSS97 fit. The CLI writes the rejection diagnostics to the dynamic JSON output and then raises `ValueError`.

### `--min-modes`

Minimum number of consecutive integer q modes required in an accepted dynamic range.

There is no default when dynamic selection is enabled. This must be provided explicitly.

### `--slope-tolerance`

Maximum allowed absolute deviation of the fitted log-log slope from -3.

For example:

```text
--slope-tolerance 0.2
```

accepts slopes between -3.2 and -2.8, provided the RMSE criterion is also satisfied.

There is no default when dynamic selection is enabled.

### `--max-log-rmse`

Maximum allowed root-mean-square residual to the best-amplitude fixed q^-3 model in natural-log space.

There is no default when dynamic selection is enabled.

The three dynamic acceptance parameters are intentionally explicit because appropriate thresholds are analysis decisions rather than universal constants. Both tolerance values must be finite and non-negative.

---

## Physical Fit Parameters

### `--lmax`

Maximum spherical harmonic index used when evaluating the theoretical fluctuation spectrum.

```bash
edgemod sample.npy --lmax 500
```

Default:

```text
500
```

### `--fixed-sigma`

By default, EdgeMod fits both bending modulus and reduced surface tension. Use:

```bash
edgemod sample.npy --fixed-sigma
```

to disable free-sigma fitting.

### `--temperature`

Experimental temperature in Kelvin, used when converting the fitted reduced tension into surface tension. Temperature must be finite and positive.

```bash
edgemod sample.npy --temperature 310
```

Default:

```text
295
```

---

## File Selection Options

### Single file

```bash
edgemod sample.npy
```

### Directory

```bash
edgemod ./edges
```

When `INPUT_PATH` is a directory, EdgeMod processes files matching:

```text
*.npy
```

### Recursive directory search

```bash
edgemod ./edges --recursive
```

With `--recursive`, EdgeMod processes files matching:

```text
**/*.npy
```

---

## Output Files

For an input file:

```text
sample.npy
```

fixed fitting writes:

```text
sample.json
sample.spectrum_diagnostic.png
```

and dynamic fitting writes:
>>>>>>> main

```text
sample.dynamic.json
```

This naming allows the same contour file to be analyzed with fixed and dynamic fitting without the second CLI run overwriting the first.

### JSON Output

The JSON contains the spectrum, latest successful fit values, and retained fit records. Each retained `SpectrumFit` records:

* fitted `kC`
* fitted surface tension
* actual lower-inclusive and upper-exclusive q bounds used for the physical fit
* the `SpectrumFitConfig` used for that fit
* the range-selector type and parameters
* range-selection diagnostics

If dynamic range selection rejects the spectrum, no new `SpectrumFit` is created. The CLI still writes `sample.dynamic.json` before re-raising the error. In that failure JSON, the top-level `fit_range_selection` field contains the best rejected range when one was evaluable, fitted slope, log-space RMSE, `accepted: false`, and the rejection reason. For a fresh CLI `Spectrum`, `kC` and `surface_tension` remain `null` because no physical fit succeeded.

---

## Python API

Scientific fitting parameters are grouped in `SpectrumFitConfig` rather than passed individually to `Spectrum.extract_kc_from_fit()`.

### Fixed fitting

```python
from vesmod.EdgeMod import (
    FixedFitRangeSelector,
    Spectrum,
    SpectrumFitConfig,
)

spectrum = Spectrum("sample.npy")
fixed_config = SpectrumFitConfig(
    lmax=500,
    free_sigma=True,
    temperature=295.0,
    range_selector=FixedFitRangeSelector(
        lower_bound=3,
        upper_bound=8,
    ),
)

fixed_fit = spectrum.extract_kc_from_fit(fixed_config)
print(fixed_fit.kC, fixed_fit.surface_tension)
```

Calling `extract_kc_from_fit()` without a config is equivalent to the historical default fixed fit:

```python
fit = spectrum.extract_kc_from_fit()
```

### Dynamic fitting

```python
from vesmod.EdgeMod import (
    QMinusThreeFitRangeSelector,
    SpectrumFitConfig,
)

dynamic_config = SpectrumFitConfig(
    range_selector=QMinusThreeFitRangeSelector(
        lower_bound=3,
        upper_bound=20,
        min_modes=5,
        slope_tolerance=0.2,
        max_log_rmse=0.1,
    ),
)

dynamic_fit = spectrum.extract_kc_from_fit(dynamic_config)
```

### Compare fixed and dynamic fits

The preceding snippets define `fixed_config` and `dynamic_config`, so both strategies can be run on the same `Spectrum`:

```python
fixed_fit = spectrum.extract_kc_from_fit(fixed_config)
dynamic_fit = spectrum.extract_kc_from_fit(dynamic_config)

print(fixed_fit.kC)
print(dynamic_fit.kC)
print(spectrum.fit_results)
```

`Spectrum.kC` and `Spectrum.surface_tension` are compatibility attributes containing the most recent successful physical fit values. A later range-selection rejection updates `Spectrum.fit_range_selection` but does not erase those successful values. The durable per-fit record is `SpectrumFit`.

---

## Example Workflows

### Analyze a single vesicle with fixed bounds

```bash
edgemod sample.npy
```

### Analyze the same vesicle with dynamic range selection

```bash
edgemod sample.npy \
    --dynamic-range \
    --lower-fitting-bound 3 \
    --upper-fitting-bound 20 \
    --min-modes 5 \
    --slope-tolerance 0.2 \
    --max-log-rmse 0.1
```

After both successful commands, both files are available:

```text
sample.json
sample.dynamic.json
```

If dynamic selection rejects the spectrum, `sample.dynamic.json` is still written with the rejection diagnostics before the CLI reports the error.

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

### Dynamic range selection requires ...

When `--dynamic-range` is supplied, all of the following must also be supplied:

```text
--min-modes
--slope-tolerance
--max-log-rmse
```

### `No trusted q range satisfied the q^-3 scaling criteria.`

The spectrum did not contain a contiguous q range inside the trusted search interval that met both dynamic acceptance criteria. The dynamic JSON output contains the selection diagnostics even though no physical fit was performed. Possible responses include:

* inspect the spectrum for whether a q^-3 regime is present at all;
* confirm that the trusted q interval is scientifically appropriate;
* inspect whether the apparent q^-3 region is too short to satisfy `--min-modes`;
* evaluate the chosen tolerances using representative accepted and rejected spectra.

Do not automatically expand the search into untrusted high-q or very-low-q regions simply to force acceptance.

### Fits produce unexpected values

Potential causes include:

* poor contour quality in the input `.npy` file
* an inappropriate fixed fitting range
* a dynamic selector accepting an unintended q region
* too few accepted contour measurements
* temperature values inconsistent with the experiment
* sensitivity of the result to the selected VesEdge QC configuration

Inspect the contour data, the spectrum, the selected q bounds, and `qc_summary.csv`. When appropriate, compare fixed and dynamic fits on the same spectrum.

---

## Citation

If EdgeMod contributes to a publication, please cite the associated manuscript and software repository.
