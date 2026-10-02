# EdgeMod spectrum plotting

`SpectrumPlotConfig` omits the translational q=1 mode by default. To include
the measured q=1 point for visual inspection, enable it explicitly:

```python
from vesmod.EdgeMod import SpectrumPlotConfig

plot_config = SpectrumPlotConfig(include_q1=True)
spectrum.plot(ax=ax, plot_config=plot_config)
```

This option affects plotting only. It does not change the fitting interval or
the fitted kC and sigma values. The default remains q >= 2 for compatibility
with existing analyses.

## Default ensemble figures

Use these helpers whenever showing an ensemble average. They reproduce the
DOPC/C16 reference style without explanatory footer text: log-log axes,
q ticks at 2/3/5/10/20, translucent gray replica lines, blue mean ± replica SEM,
orange solid fixed-zero-sigma fits, and purple dashed free-sigma fits.

```python
from vesmod.EdgeMod import EnsemblePlotConfig, plot_ensemble_spectra

# Set this to the actual power definition used to construct the spectra.
style = EnsemblePlotConfig(power_definition="temporal_variance")
fig, ax = ensemble.plot(label="DOPC", config=style)
fig.savefig("dopc_spectra.pdf", bbox_inches="tight")

fig, axes = plot_ensemble_spectra(
    ensembles,
    labels=["1.25% C16", "2.5% C16", "3.75% C16", "5% C16"],
    config=style,
    title="C16 spectra and ensemble averages by concentration",
)
fig.savefig("c16_spectra.pdf", bbox_inches="tight")
```

The functions use the most recent recorded fit of each kind from
`ensemble.fit_results`. Call `ensemble.extract_kc_from_fit(...)` upstream if
fits are needed; plotting never fits, selects replicas, or changes QC. Pass
`fits=[fixed_fit, free_fit]` to select explicit records, or `fits=[]` for no
curves. The multipanel helper takes one fit sequence per panel.

Fit curves use each record's inclusive lower and exclusive upper q bounds.
Camera-aware fits use their recorded exposure time, viscosities, and replica
radii. Values are measured powers; plotting does not remove camera averaging
or apply q³ scaling. Mean-square inputs are labeled “Mean-square power”;
temporal-variance inputs must explicitly use the configuration shown above.
All replicas contribute equally to the mean and sample SEM, regardless of
the weighting used for fitting. For one replica, no SEM is drawn.

Panel titles show composition, n, kC in kBT, and free-fit sigma. SI sigma is
evaluated at the mean replica radius using the fit's temperature, with an
explicit “mean radius” label. It is a display convention, not a common
physical tension across replicas. If radii are unavailable, the title reports
reduced sigma instead.

Default limits are `xlim=(1.8, 22.5)`, `ylim=(5e-8, 5e-3)`; every panel uses
the same limits. Override these through `EnsemblePlotConfig` when needed.
The default grid has four columns, adjustable with `ncols`. Single panels
place fitted values in the lower-left legend; multiple panels put them in
panel titles with a shared legend above. Both helpers return matplotlib
objects and leave saving/closing to the caller.

