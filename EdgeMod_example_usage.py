"""
Example usage for EdgeMod.

Update fpath and run this file, or adapt for your own use.
"""
from pathlib import Path
import glob
from vesmod.EdgeMod import Spectrum

fpath = "YOUR/PATH/HERE/"  # path to the directory containing your .npy file(s)
n_theta = 120              # Number of angular bins to downsample to
lower_fitting_bound = 3    # Lowest mode to fit to
upper_fitting_bound = 8    # Ignore this mode and above when fitting
lmax = 500                 # Maximum summation index
free_sigma = True          # Fit surface tension (sigma) in addition to kc

for file in glob.glob(fpath + '*.npy', recursive=True):
    path = Path(file).resolve()
    print(f"working on file {path.stem}")
    spectrum = Spectrum(file, Ntheta=n_theta)
    kc, sigma = spectrum.extract_kc_from_fit(
        lower_bound=lower_fitting_bound,
        upper_bound=upper_fitting_bound,
        lmax=lmax,
        free_sigma=free_sigma,
    )
    print(kc, sigma)
    spectrum.to_json(path)
