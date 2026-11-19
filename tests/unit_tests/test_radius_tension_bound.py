"""Verify radius bounds reach lmfit across public fitting paths."""
from types import SimpleNamespace

import numpy as np
import pytest

from vesmod.edgemod import EnsembleMember
from vesmod.edgemod import Spectrum, SpectrumEnsemble, SpectrumFitConfig
from vesmod.edgemod.fitting.fitting_functions.spherical_model import configure_spectrum_model


@pytest.mark.parametrize('exposure,chi', [(0, None), (.030, None), (.030, 4.1e-9)])
@pytest.mark.parametrize('free_sigma', [False, True])
def test_public_single_and_ensemble_bounds(monkeypatch, exposure, chi, free_sigma):
    captured = []

    def capture(self, data, **kwargs):
        params = kwargs['params']
        captured.append(params['sigma'].max)
        return SimpleNamespace(best_values={'kC': 25, 'sigma': 0},
                               best_fit=np.asarray(data), chisqr=0, redchi=0)

    monkeypatch.setattr('vesmod.edgemod.fitting.optimization.least_squares.Model.fit', capture)
    config = SpectrumFitConfig(camera_integration=exposure > 0, relaxation_model="solvent_and_membrane" if chi is not None else "solvent_only", lmax=20, free_sigma=free_sigma,
                               exposure_time=exposure, eta_m=chi,
                               viscosity_in=.001, viscosity_out=.001)
    ensemble = SpectrumEnsemble()
    for radius in [7.0, 2.0, 5.0]:
        spectrum = Spectrum.from_radii(np.full((2, 32), radius))
        spectrum.extract_kc_from_fit(config)
        ensemble.add_spectrum(EnsembleMember(spectrum.powers, spectrum.modes, 25, r0=radius))
    fit = ensemble.extract_kc_from_fit(config)
    assert captured == pytest.approx([4900, 400, 2500, 100 * (14 / 3)**2])
    assert fit.to_dict()['radii'] == [7, 2, 5]


def test_ensemble_member_requires_radius():
    with pytest.raises(TypeError, match="r0"):
        EnsembleMember(np.ones(5), range(3, 8))


@pytest.mark.parametrize("radius", [None, 0, -1, float("nan"), float("inf")])
def test_ensemble_member_rejects_invalid_radius(radius):
    with pytest.raises((TypeError, ValueError), match="reference_radius"):
        EnsembleMember(np.ones(5), range(3, 8), r0=radius)


def test_direct_fit_requires_radii_without_exposure():
    with pytest.raises(ValueError, match='radii are required'):
        configure_spectrum_model(SpectrumFitConfig(), None)
    with pytest.raises(ValueError, match='expected 2 radii'):
        configure_spectrum_model(SpectrumFitConfig(), [5], expected_replica_count=2)


def test_configure_spectrum_model_requires_config():
    with pytest.raises(TypeError, match="config must be a SpectrumFitConfig"):
        configure_spectrum_model(None, [5])
