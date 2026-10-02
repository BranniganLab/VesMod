# Ensemble spectrum figures

When plotting spectra with an ensemble average, use
`SpectrumEnsemble.plot()` or `plot_ensemble_spectrum()` for one panel and
`plot_ensemble_spectra()` for multiple panels. Import these functions from
`vesmod.EdgeMod`. See `docs/EdgeMod_spectrum_plotting.md` for examples.

These helpers define the default visual style: log-log axes, faint gray
replica traces, blue equal-replica means with SEM, orange solid fixed-zero-sigma
fits, purple dashed free-sigma fits, and composition/count/fit panel titles.
Do not recreate this style in ad hoc plotting scripts. Do not add explanatory
footer text or reserve space for it unless explicitly requested.

Plotting must not refit data, change selection/QC, or choose scientific fitting
settings. Use the recorded fit configuration and radius provenance. Specify
the actual power definition when plotting temporal variance.
