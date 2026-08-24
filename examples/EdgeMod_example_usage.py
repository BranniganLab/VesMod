"""
Example usage for EdgeMod.

Update runtime parameters and run this file, or adapt for your own use.
"""
from pathlib import Path
import glob
from vesmod.EdgeMod import (
    FixedFitRangeSelector,
    Spectrum,
    SpectrumFitConfig,
)

fpath = "YOUR/PATH/HERE/"  # path to the directory containing your .npy file(s)
config = SpectrumFitConfig(
    lmax=500,
    free_sigma=True,
    temperature=295.0,
    range_selector=FixedFitRangeSelector(
        lower_bound=3,
        upper_bound=8,
    ),
)

for file in glob.glob(fpath + '*.npy', recursive=True):
    path = Path(file).resolve()
    print(f"working on file {path.stem}")
    spectrum = Spectrum(file)
    fit = spectrum.extract_kc_from_fit(config)
    print(fit.kC, fit.surface_tension)
    spectrum.to_json(path)
