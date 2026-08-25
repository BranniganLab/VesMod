"""Example usage for the stable core EdgeMod physical fitter."""

from pathlib import Path
import glob

from vesmod.EdgeMod import Spectrum, SpectrumFitConfig

fpath = "YOUR/PATH/HERE/"  # directory containing .npy file(s)
config = SpectrumFitConfig(
    lower_bound=3,
    upper_bound=8,
    lmax=500,
    free_sigma=True,
    temperature=295.0,
)

for file in glob.glob(fpath + "*.npy", recursive=True):
    path = Path(file).resolve()
    print(f"working on file {path.stem}")
    spectrum = Spectrum(file)
    fit = spectrum.extract_kc_from_fit(config)
    print(fit.kC, fit.surface_tension)
    spectrum.to_json(path)
