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

EdgeMod stores returned optimizer results and writes diagnostic PNGs without applying post-fit acceptance criteria. Input and model-domain errors still propagate.

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

## Citation

If EdgeMod contributes to a publication, please cite the associated manuscript and software repository.

### Camera integration time

Enable finite-exposure fitting with an integration time in **seconds**:

```bash
edgemod contours.npy --exposure-time 0.030 --viscosity-in 0.00102 --viscosity-out 0.00097
```

The default is `--exposure-time 0`, which preserves the instantaneous HSS97
fit. Both solvent viscosities must be supplied explicitly, in **Pa s**.
For example, an experiment with interior viscosity **1.02 mPa s** and exterior
viscosity **0.97 mPa s** uses:

```bash
edgemod contours.npy --exposure-time 0.030 --viscosity-in 0.00102 --viscosity-out 0.00097
```

In Python:

```python
config = SpectrumFitConfig(exposure_time=0.030, viscosity_in=0.00102, viscosity_out=0.00097)
fit = spectrum.extract_kc_from_fit(config)
```

Each spherical-mode contribution `Nlq Plq²` is multiplied by
`B(x) = 2(x - 1 + exp(-x))/x²`, with `x = exposure_time / tau_l`, before
summing the contour spectrum. The solvent-only spherical relaxation rate is
from Faizi et al., *Soft Matter* (2020), equation 2
([DOI: 10.1039/d0sm00943a](https://doi.org/10.1039/d0sm00943a)). The boxcar
averaging factor follows integration of an exponential autocorrelation and
matches the finite-exposure factor in Kumar et al., *Soft Matter* (2020),
equation 6 ([DOI: 10.1039/c9sm02048a](https://doi.org/10.1039/c9sm02048a)).
Relaxation times update with fitted kC and reduced tension during optimization.
The mean radius in microns is converted to meters, and kC in kBT to joules.
Corrected fits require reduced tension greater than -6 for stable spherical
modes; the optimizer enforces this bound. `--fixed-sigma` still fixes tension
to zero.

Ensemble fits use free sigma by default. Supply
`SpectrumFitConfig(free_sigma=False, ...)` to fix reduced tension to zero.
For exposure-aware ensemble fits, supply the radius when adding **each** replica:

```python
ensemble.add_spectrum(spectrum.avg_amps2, spectrum.modes, spectrum.kC, r0=spectrum.r0)
fit = ensemble.extract_kc_from_fit(
    SpectrumFitConfig(exposure_time=0.030,
                      viscosity_in=0.00102, viscosity_out=0.00097),
    weight_by_replica_sem=True,
)
```

The ensemble model averages predictions at the individual radii with shared
kC and reduced tension, using the same equal replica weights as the measured
mean. Missing radii raise an error for every fit. Exposure,
viscosities, and temperature are retained in fit configuration; corrected
ensemble records also retain the replica radii.

This model assumes the detected contour approximates an exposure-averaged
radial displacement, exponential spherical-mode relaxation, and dissipation
in the interior and exterior solvents. It does not include membrane viscosity
or image formation effects. Exposure time is the integration duration per
frame, rather than the interval between frames. The correction changes the
fitted theoretical prediction; measured Fourier powers remain as calculated.

## Selecting the Fourier power definition

The `Spectrum` Python API supports two definitions of Fourier-mode power:

* `mean_square` (default): the time average of `|u_q(t)|**2`, matching the
  existing EdgeMod calculation.
* `temporal_variance`: the time average of
  `|u_q(t) - mean_t(u_q(t))|**2`, subtracting the complex temporal mean as in
  Gracia et al. (Soft Matter, 2010, equations 7-8).

Choose the definition when constructing a spectrum. The selected power is
stored in `avg_amps2` and used by subsequent fits. Here, `edges` is a QCed
`VesicleEdges` object or a path to the saved radii array.

```python
from vesmod.EdgeMod import Spectrum

mean_square = Spectrum(edges, power_definition="mean_square")
mean_square_fit = mean_square.extract_kc_from_fit()

temporal_variance = Spectrum(edges, power_definition="temporal_variance")
temporal_variance_fit = temporal_variance.extract_kc_from_fit()
```

Each `SpectrumFit` and exported spectrum records the selected definition. To
isolate its effect, fit both spectra with the same `SpectrumFitConfig`.

`temporal_variance` requires at least two frames. It removes a mode's mean in
the recorded angular coordinate system; it does not distinguish thermal
fluctuations from a persistent deformation that changes orientation during
recording. This option is available through the Python API, not the CLI.

### Display camera-corrected powers in Python

After fitting a `Spectrum` with nonzero `exposure_time` and explicit viscosities:

```python
from vesmod.EdgeMod.spectrum_plotting import SpectrumPlotConfig

spectrum.extract_kc_from_fit(fit_config)
result = spectrum.plot(plot_config=SpectrumPlotConfig(power="both"))
result.ax.legend()
```

`power="measured"` (the default) shows the measured powers and camera-averaged
fit. `power="corrected"` shows only the corrected powers and instantaneous
HSS97 curve. `power="both"` overlays both: circles for measured powers and
squares for corrected powers, with solid and dashed theory curves respectively.
The same option is supported by `plot_q3_scaled_spectrum`.

Corrected powers are the measured powers multiplied by the fitted ratio
`instantaneous_power / camera_power`. They depend on the fitted parameters and
are intended for display; stored powers and fits remain unchanged. Correction
requires an existing camera integration fit, excludes modes above `lmax`, and
cannot be combined with `include_q1=True`. Nonfit modes are shown faded as usual;
their correction extrapolates the fitted model.

### Fit diagnostics

Individual spectra, ensembles, and the tuple-returning fitting helper retain
returned optimizer estimates without checking convergence status, parameter-bound
proximity, uncertainty, or `kC`–`sigma` correlation. Relative RMSE is logged for
each fit; uncertainties and residuals remain visible in diagnostic plots.
Optimizer parameter bounds, input validation, and optional experimental q-range
selection still apply.

### Membrane viscosity in camera averaging (Faizi 2024)

Supply `--eta-m` to select Equation 1 of
[Faizi, Granek, and Vlahovska (2024)](https://www.pnas.org/doi/10.1073/pnas.2413557121):

```bash
edgemod contours.npy --exposure-time 0.030 --viscosity-out 0.001 --eta-m 4.1e-9
```

The Python equivalent is
`SpectrumFitConfig(exposure_time=0.030, viscosity_out=0.001, eta_m=4.1e-9)`.
`eta_m` is a fixed, finite, nonnegative membrane surface viscosity in **Pa s m**,
not a fitted parameter. It is saved with the fit configuration. Omitting it
retains the solvent-only rate; supplying zero selects the nonviscous limit.

Each replica uses `chi_s = eta_m/(R0_meters*viscosity_out)`. Thus ensembles
share the dimensional membrane viscosity while using each replica's own radius
and chi_s in the exposure correction. The mean radius is used only for the
ensemble reduced-tension upper bound, not the relaxation rates.

Both rates have the numerator
`kappa/(eta_out R^3) * (l-1) l (l+1) (l+2) * (l(l+1)+sigma_reduced)`.
The solvent-only denominator is
`4l^3+6l^2-1 + (2l^3+3l^2-5)*(eta_in/eta_out-1)`;
Equation 1 uses
`4l^3+6l^2-1 + (4l^2+4l-8)*chi_s`.
Equation 1 uses one common solvent viscosity: `viscosity_out` supplies eta,
and `viscosity_in` must be omitted or equal to it.

Here `kappa = kC*k_B*T`, radii are converted from microns to meters, and rates
are in inverse seconds. Positive eta_m slows relaxation and retains more power
during camera exposure. The zero-exposure equilibrium spectrum is unchanged.

The paper infers viscosity from contour autocorrelations. EdgeMod uses its
predicted spherical-mode rates in the exposure integral: each spherical l
contribution is multiplied by `B(t_exp*omega_l)` before projection onto contour
q, rather than treating the entire contour mode as a single exponential.

### Radius-dependent reduced-tension bound

Every fit now uses `sigma_reduced <= 100 * R0**2`, with `R0` in **microns**.
For a single vesicle, `R0` is its mean radius. This applies with and without
camera averaging, for both relaxation models and both fixed/free sigma fits.
Direct fitting helpers require `radii`; ensembles require `r0` when adding
**every** replica, even without camera averaging. The example for combining
replicas reads this value from each spectrum's saved metadata.

Ensembles fit one shared reduced tension, so their upper bound is
`100 * mean(replica_radii)**2`, using the arithmetic mean of the replica
mean radii, then squaring it. Replica radii are retained in ensemble fit
metadata.

Since physical tension is `sigma_reduced * kC * k_B*T / R0_meters**2`,
the proposed rule limits `physical_tension/kappa` to `100 / micron**2`
(equivalently `1e14 / meter**2`). It is not a fixed physical tension cap:
the allowed tension still depends on the fitted bending modulus.
