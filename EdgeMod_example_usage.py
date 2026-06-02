"""
Example usage for EdgeMod.

Update fpath and run this file, or adapt for your own use.
"""
from pathlib import Path
import glob
from vesmod.EdgeMod import Spectrum

fpath = "YOUR/PATH/HERE/"  # path to the directory containing your .nd2 file(s)

for file in glob.glob(fpath + '*.npy', recursive=True):
    path = Path(file).resolve()
    print(f"working on file {path.stem}")
    spectrum = Spectrum(file, Ntheta=120)
    kc, sigma = spectrum.extract_kc_from_fit(
        lower_bound=3,
        upper_bound=8,
        lmax=500,
        free_sigma=True,
    )
    print(kc, sigma)
    spectrum.to_json(path)
