"""Verify radius bounds reach lmfit across public fitting paths."""
from types import SimpleNamespace

import numpy as np
import pytest

from vesmod.EdgeMod import Spectrum, SpectrumEnsemble, SpectrumFitConfig
from vesmod.EdgeMod.spectrum_utils import MiniSpectrum, fit_spectrum_lmfit


@pytest.mark.parametrize('exposure,chi', [(0, None), (.030, None), (.030, 8)])
@pytest.mark.parametrize('free_sigma', [False, True])
def test_public_single_and_ensemble_bounds(monkeypatch, exposure, chi, free_sigma):
    captured = []

    def capture(self, data, **kwargs):
        params = kwargs['params']
        captured.append(params['sigma'].max)
        return SimpleNamespace(best_values={'kC': 25, 'sigma': 0},
                               best_fit=np.asarray(data), chisqr=0, redchi=0)

    monkeypatch.setattr('vesmod.EdgeMod.spectrum_utils.Model.fit', capture)
    config = SpectrumFitConfig(lmax=20, free_sigma=free_sigma,
                               exposure_time=exposure, chi_s=chi,
                               viscosity_in=.001, viscosity_out=.001)
    ensemble = SpectrumEnsemble()
    for radius in [7.0, 2.0, 5.0]:
        spectrum = Spectrum.from_radii(np.full((2, 32), radius))
        spectrum.extract_kc_from_fit(config)
        ensemble.add_spectrum(spectrum.avg_amps2, spectrum.modes, 25, r0=radius)
    fit = ensemble.extract_kc_from_fit(config)
    assert captured == pytest.approx([4900, 400, 2500, 100 * (14 / 3)**2])
    assert fit.to_dict()['radii'] == [7, 2, 5]


def test_ensemble_missing_radius_fails_without_exposure():
    ensemble = SpectrumEnsemble()
    ensemble.add_spectrum(np.ones(5), range(3, 8), 25, r0=5)
    ensemble.add_spectrum(np.ones(5), range(3, 8), 25)
    with pytest.raises(ValueError, match='r0 for every replica'):
        ensemble.extract_kc_from_fit()


def test_direct_fit_requires_radii_without_exposure():
    group = MiniSpectrum(np.arange(3, 8), np.ones(5), None)
    with pytest.raises(ValueError, match='radii are required'):
        fit_spectrum_lmfit(group, 20)
    with pytest.raises(ValueError, match='expected 2 radii'):
        fit_spectrum_lmfit(group, 20, radii=[5], expected_replica_count=2)
