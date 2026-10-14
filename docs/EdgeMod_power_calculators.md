# Replaceable spectrum power calculation

`Spectrum` accepts a callable calculator in every construction route. Omission defaults to `MeanSquaredPowerCalculator`, preserving the historical calculation.

```python
from vesmod.EdgeMod import Spectrum, TemporalVarianceCalculator

spectrum = Spectrum.from_radii(radii, calculator=TemporalVarianceCalculator())
print(spectrum.modes, spectrum.powers)
```

`TemporalVarianceCalculator` subtracts each normalized complex Fourier coefficient's temporal mean before averaging squared magnitudes. `MeanSquaredPowerCalculator` does not subtract it. Both divide the sum over frames by the number of retained frames. They preserve the existing reference radius, dimensionless normalization and full signed FFT modes.

## Custom functions

```python
import numpy as np
from vesmod.EdgeMod import Spectrum, PowerCalculationResult


def custom_calculator(radii):
    radius = float(np.mean(radii))
    coefficients = np.fft.fft(radii, axis=1) / (radius * radii.shape[1])
    return PowerCalculationResult(
        modes=np.rint(np.fft.fftfreq(radii.shape[1]) * radii.shape[1]).astype(int),
        powers=np.mean(np.abs(coefficients) ** 2, axis=0),
        reference_radius=radius,
        radius_unit="um",
        power_definition="custom_mean_square",
        power_unit="dimensionless",
    )

spectrum = Spectrum.from_radii(radii, calculator=custom_calculator)
```

Input radii are positive finite real values in micrometers, with shape `(frames, angular_samples)` on a uniform periodic angular grid without a duplicated endpoint. Spectrum validates and applies `frame_cutoff` before invoking the calculator. Custom errors propagate, with no fallback.

Results require unique integer modes, matching finite nonnegative powers, a positive reference radius, units and a power definition. Zero power is valid. Spectrum exposes `modes`, `powers`, `r0` and `power_definition` as read-only properties. To change the calculation, construct a new Spectrum. `r0` is the result's reference radius converted to micrometers; supported radius units are `um`, `m` and `nm`.

There are no normalization, calculator settings, calculator identity or calculator version fields. Document custom conventions in the calculator's API. Existing physical routines still expect the historical dimensionless observable; metadata is not automatic adaptation of custom powers.

## Breaking migration

- Replace `Spectrum(..., power_definition="temporal_variance")` with `Spectrum(..., calculator=TemporalVarianceCalculator())`.
- Replace `power_definition="mean_square"` with `calculator=MeanSquaredPowerCalculator()` or omit it.
- Replace `spectrum.avg_amps2` with `spectrum.powers`.
- Calculated arrays can be mutated in place, while their properties and metadata cannot be reassigned. Use a calculator to supply custom data during construction.
- Exported Spectrum JSON uses `powers` instead of `avg_amps2`, with `power_definition`, `power_unit`, `reference_radius` and its `radius_unit` (with `r0` always in micrometers). There is no legacy alias or automatic migration reader. Existing JSON consumers must rename the key; `.npy` contour inputs are unchanged.

`SpectrumEnsemble`, `MiniSpectrum` and plotting data records retain their existing attribute names in this component-only refactor. Pass `spectrum.powers` to their existing interfaces.
