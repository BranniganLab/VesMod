# EdgeMod CLI

EdgeMod fits membrane mechanical parameters from vesicle contour trajectories. It reads NumPy `.npy` edge files, computes fluctuation spectra, fits a configured Fourier-mode interval to the HSS97 physical model, and writes JSON results for downstream analysis.

The **stable core EdgeMod API always fits explicit q bounds**. The default is the historical lower-inclusive, upper-exclusive interval `3 <= q < 8` (q = 3, 4, 5, 6, 7).

Dynamic q-range selection is an **experimental feature**. It lives under `vesmod.EdgeMod.experimental` and is composed upstream of the stable physical fitter. The experimental selector chooses q bounds; the core `Spectrum` fitter does not know how those bounds were chosen.

## Features

Stable core features:

* Process a single `.npy` file or a directory of `.npy` files
* Optional recursive directory traversal
* Optional separate output tree with preserved relative input paths
* Batch provenance and one summary row per attempted fit
* Automatic fluctuation spectrum calculation
* Fitting of membrane bending modulus ($k_C$)
* Optional fitting of membrane surface tension ($\sigma$)
* Explicit lower/upper Fourier fitting bounds
* Configurable spherical-harmonic summation limit
* Spectrum-fit diagnostic PNG for attempted physical fits
* JSON output containing fit values, q bounds, and physical-fit configuration

Experimental features:

* Optional q^-3-based dynamic range selection inside a trusted q interval
* Explicit slope/RMSE acceptance criteria
* Rejection when no trustworthy q^-3 regime is found
* Separate dynamic JSON output containing experimental selection diagnostics
* Separate temporal-RMS screening stage with batch provenance and diagnostics

---

## Quick Start

Fit one contour trajectory using the stable historical defaults:

```bash
edgemod "sample.npy"
```

Defaults:

```text
q = 3, 4, 5, 6, 7
lmax = 500
free sigma = True
temperature = 295 K
```

Successful fixed fitting writes:

```text
sample.json
sample.spectrum_diagnostic.png
```

If physical-fit validation fails after HSS97 fitting is attempted, EdgeMod writes the diagnostic PNG before propagating the validation error.

For a VesEdge batch with each analysis stage in its own directory:

```bash
vesedge qc "./checkpoints" --output-dir ./results/qc_standard
edgemod "./results/qc_standard" \
    --recursive \
    --output-dir ./results/edgemod_standard
```

---

## Input Requirements

EdgeMod operates on contour arrays stored in `.npy` files. Each row is one accepted frame and each column is one angular sample:

```python
(n_frames, n_theta)
```

Distances should be in microns. `vesedge qc` output is directly suitable for EdgeMod.

---

## Stable Fixed q-Range Fitting

Fixed fitting is the default and is the stable EdgeMod behavior.

```bash
edgemod "sample.npy" \
    --lower-fitting-bound 3 \
    --upper-fitting-bound 8
```

The lower bound is inclusive and the upper bound is exclusive.

### `--lower-fitting-bound`

Lowest Fourier mode used by the physical fit. Default: `3`.

### `--upper-fitting-bound`

First Fourier mode excluded from the physical fit. Default: `8`.

### `--lmax`

Maximum spherical harmonic index used when evaluating the theoretical fluctuation spectrum. Default: `500`.

### `--fixed-sigma`

By default, EdgeMod fits reduced surface tension as a free parameter. Use `--fixed-sigma` to hold it fixed.

### `--temperature`

Experimental temperature in Kelvin used to convert fitted reduced tension into physical surface tension. It must be finite and positive. Default: `295`.

---

## Output Directory and Batch Provenance

By default, EdgeMod remains backward compatible and writes each JSON result and
diagnostic beside its input `.npy`. Use `--output-dir` to keep fitting
artifacts in a separate tree:

```bash
edgemod "./results/qc_standard" \
    --recursive \
    --output-dir ./results/edgemod_standard
```

Relative input directories are preserved. For example,
`qc_standard/condition_a/sample.npy` produces:

```text
edgemod_standard/
├── condition_a/
│   ├── sample.json
│   └── sample.spectrum_diagnostic.png
├── edgemod_fit.json
└── fit_summary.csv
```

`edgemod_fit.json` records the resolved input manifest, recursion setting,
physical fit configuration, optional dynamic-range configuration, and the
artifacts managed by the batch. `fit_summary.csv` contains one row per
attempted input with its status, fitted values when available, and any error.

External input and output paths must not overlap. Reusing an output directory
with a different input selection or configuration is rejected unless
`--overwrite` is supplied. Overwrite cleanup removes only JSON and diagnostic
PNG files recorded in the preceding valid artifact manifest; unrelated files
are preserved.

When a compatible result already exists and `--overwrite` is omitted, EdgeMod
keeps it and records `kept_existing` in the new summary. Omit
`--output-dir` to retain the historical beside-input behavior.

---

## Experimental Dynamic q-Range Selection

Dynamic selection is explicitly experimental and is only enabled with:

```bash
--dynamic-range
```

The CLI then performs two separate operations:

```text
experimental q^-3 selector
        ↓ selected q bounds
SpectrumFitConfig
        ↓
stable HSS97 physical fit
```

The selector searches only within the interval defined by `--lower-fitting-bound` and `--upper-fitting-bound`; it does not expand into untrusted lower- or higher-q regions.

Example:

```bash
edgemod "sample.npy" \
    --dynamic-range \
    --lower-fitting-bound 3 \
    --upper-fitting-bound 20 \
    --min-modes 5 \
    --slope-tolerance 0.2 \
    --max-log-rmse 0.1
```

Candidate contiguous integer-q windows are evaluated in log space using both:

1. deviation of the fitted power-law slope from -3; and
2. RMSE to the best-amplitude fixed q^-3 model.

Among accepted candidates, the longest range is preferred. If none passes, the experimental selector rejects the spectrum before HSS97 fitting.

### `--min-modes`

Minimum number of consecutive q modes required for an accepted experimental range. No default is supplied in dynamic mode.

### `--slope-tolerance`

Maximum allowed absolute deviation of the fitted log-log slope from -3. Must be supplied explicitly and be finite/non-negative.

### `--max-log-rmse`

Maximum allowed natural-log-space RMSE to the fixed q^-3 model. Must be supplied explicitly and be finite/non-negative.

---

## Experimental Temporal-RMS Screening

Temporal RMS is a separate optional stage, analogous to `vesedge qc`. It does
not modify `Spectrum` or the core physical fit.

List the experimental commands with:

```bash
edgemod experimental --help
```

Measure every selected trajectory without excluding any input:

```bash
edgemod experimental temporal-rms "./results/qc_standard" \
    --output-dir ./results/rms_report
```

Apply an explicitly chosen cutoff and export only accepted trajectories:

```bash
edgemod experimental temporal-rms "./results/qc_standard" \
    --output-dir ./results/rms_50nm \
    --cutoff-nm 50

edgemod "./results/rms_50nm"
```

The stage removes each Fourier mode's temporal mean before combining its power,
so persistent noncircularity does not count as motion. Input distances must be
in microns; reported amplitudes and `--cutoff-nm` are in nanometers. The default
mode interval is lower-inclusive and upper-exclusive, `3 <= q < 8`, and can be
changed with `--lower-bound` and `--upper-bound`.

Relevant options:

* `--recursive`: search input subdirectories recursively
* `--lower-bound`: first included Fourier mode; default `3`
* `--upper-bound`: first excluded Fourier mode; default `8`
* `--cutoff-nm`: optional minimum included amplitude in nanometers
* `--overwrite`: replace outputs from an incompatible prior screening run

Without `--cutoff-nm`, every successfully measured trajectory is exported. With a cutoff, below-threshold trajectories remain in the CSV and histogram but are not copied into the accepted output set.

For an input directory containing `sample.npy`, the output directory contains:

```text
temporal_rms_qc.json
temporal_rms_summary.csv
temporal_rms_histogram.png
sample.npy  # only when included
```

Relative input paths are preserved during recursive processing. The input and output paths must not overlap, which prevents screening exports from being mistaken for new inputs or overwriting source arrays.

As with `vesedge qc`, incompatible existing provenance requires another output directory or `--overwrite`. Temporal RMS is experimental and should not be treated as a universal physical criterion without empirical calibration.

---

## Output Files

For `sample.npy`, stable fixed fitting writes:

```text
sample.json
sample.spectrum_diagnostic.png
```

Successful experimental dynamic selection followed by a successful physical fit writes:

```text
sample.dynamic.json
sample.dynamic.spectrum_diagnostic.png
```

The distinct filenames allow fixed and experimental analyses to be compared without overwriting one another.

If experimental dynamic selection rejects the spectrum before physical fitting, EdgeMod writes:

```text
sample.dynamic.json
```

but no dynamic spectrum-fit PNG, because no HSS97 fit was attempted.

### JSON structure

Core `SpectrumFit` records contain only physical-fit information:

* fitted `kC`
* fitted surface tension
* lower-inclusive and upper-exclusive q bounds actually fit
* the stable `SpectrumFitConfig`

They do **not** contain range-selector type or dynamic-selection diagnostics.

When dynamic mode is used, the CLI adds a separate experimental section:

```json
{
  "experimental": {
    "dynamic_range_selection": {
      "accepted": true,
      "lower_bound": 5,
      "upper_bound": 12,
      "slope": -3.01,
      "log_rmse": 0.02,
      "reason": null
    }
  }
}
```

This keeps experimental provenance out of the stable `Spectrum` and `SpectrumFit` object model while preserving the diagnostics in CLI output.

---

## Python API

### Stable core fitting

```python
from vesmod.EdgeMod import Spectrum, SpectrumFitConfig

spectrum = Spectrum("sample.npy")
config = SpectrumFitConfig(
    lower_bound=3,
    upper_bound=8,
    lmax=500,
    free_sigma=True,
    temperature=295.0,
)

fit = spectrum.extract_kc_from_fit(config)
print(fit.kC, fit.surface_tension)
```

Calling `extract_kc_from_fit()` without a config uses the historical fixed defaults.

### Experimental dynamic selection

The dynamic selector is deliberately imported from the experimental namespace:

```python
from dataclasses import replace

from vesmod.EdgeMod import Spectrum, SpectrumFitConfig
from vesmod.EdgeMod.experimental import QMinusThreeRangeSelector

spectrum = Spectrum("sample.npy")
base_config = SpectrumFitConfig(
    lower_bound=3,
    upper_bound=20,
)
selector = QMinusThreeRangeSelector(
    lower_bound=3,
    upper_bound=20,
    min_modes=5,
    slope_tolerance=0.2,
    max_log_rmse=0.1,
)

selection = selector.select(spectrum.modes, spectrum.avg_amps2)
if not selection.accepted:
    raise ValueError(selection.reason)

dynamic_config = replace(
    base_config,
    lower_bound=selection.lower_bound,
    upper_bound=selection.upper_bound,
)
dynamic_fit = spectrum.extract_kc_from_fit(dynamic_config)
```

The dependency direction is therefore experimental selection -> fixed q bounds -> stable physical fit.

`Spectrum` does not have a `fit_range_selection` attribute, and `SpectrumFitConfig` does not have a `range_selector` attribute.

---

## File Selection and Batch Behavior

Always double-quote each input path or pattern. Quoting ordinary paths is safe,
handles spaces, and prevents the shell from expanding wildcard patterns before
EdgeMod receives them. This lets EdgeMod apply suffix validation, deduplication,
and `--recursive` consistently. Double quotes also allow shell variables such as
`"$DATA_DIR/*.npy"` to expand while preserving the wildcard for EdgeMod.

Single file:

```bash
edgemod "sample.npy"
```

Directory:

```bash
edgemod "./edges"
```

Multiple selectors:

```bash
edgemod "./condition_a/sample.npy" "./condition_b/sample.npy"
```

Wildcard pattern:

```bash
edgemod "./conditions/*/sample.npy"
```

Recursive directory search:

```bash
edgemod "./edges" --recursive
```

Recursive wildcard search:

```bash
edgemod "./conditions/*" --recursive
```

Do not remove the quotes from wildcard examples. An unquoted wildcard may be
expanded by the shell first, changing which selectors EdgeMod sees and how
recursive discovery behaves.

Recursive runs report expected fitting/numerical failures and continue with later spectra. Direct non-recursive runs propagate those failures to the caller.

---

## Troubleshooting

### Dynamic range selection requires ...

When `--dynamic-range` is supplied, all of these are required:

```text
--min-modes
--slope-tolerance
--max-log-rmse
```

### `No trusted q range satisfied the q^-3 scaling criteria.`

The experimental selector found no contiguous q interval inside the trusted search bounds that met both acceptance criteria. Inspect the dynamic JSON diagnostics, the measured spectrum, the trusted interval, and the chosen tolerances. Do not expand the search into scientifically untrusted q regions merely to force acceptance.

### Fits produce unexpected values

Potential causes include poor contour quality, an inappropriate fixed range, an experimental selector accepting an unintended range, too few accepted frames, temperature mismatch, or sensitivity to VesEdge QC settings. Compare fixed and dynamic results explicitly when evaluating the experimental method.

### EdgeMod input and output paths overlap

Choose a fit `--output-dir` outside the selected input file or directory.
External fit outputs are kept separate from the QC arrays they consume.

### EdgeMod output directory contains another batch

Choose another `--output-dir`, or use `--overwrite` after confirming that
the recorded prior EdgeMod artifacts should be replaced. Unrelated files are
not removed.

### Temporal-RMS input and output paths overlap

Choose an `--output-dir` outside the selected input file or directory. In-place screening is intentionally rejected to protect source arrays and prevent recursive rediscovery of exports.

---

## Citation

If EdgeMod contributes to a publication, please cite the associated manuscript and software repository.
