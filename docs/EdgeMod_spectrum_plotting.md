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
