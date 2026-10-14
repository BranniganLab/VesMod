"""Scientific and ownership contracts for replaceable power calculations."""



import numpy as np
import pytest

from vesmod.EdgeMod import (
    MeanSquaredPowerCalculator, TemporalVarianceCalculator,
    PowerCalculationResult, Spectrum,
)


def _result(**changes):
    values = dict(modes=np.array([0, 4]), powers=np.array([1., .02]),
                  reference_radius=10., radius_unit='um',
                  power_definition='custom', power_unit='dimensionless')
    values.update(changes)
    return PowerCalculationResult(**values)


def test_custom_callable_receives_independent_input_and_owns_output():
    radii = np.full((4, 16), 10.)
    powers = np.array([1., .02])
    observed = []

    def calculate(retained):
        observed.append(retained.shape)
        retained[0, 0] = 2
        assert radii[0, 0] == 10
        return _result(powers=powers)

    spectrum = Spectrum.from_radii(radii, frame_cutoff=2, calculator=calculate)
    radii[:] = 20
    powers[:] = 100
    assert observed == [(2, 16)]
    np.testing.assert_array_equal(spectrum.powers, [1., .02])
    assert spectrum.r0 == 10
    spectrum.powers[0] = 3
    spectrum.modes[0] = -1
    assert spectrum.powers[0] == 3
    assert spectrum.modes[0] == -1
    assert powers[0] == 100
    with pytest.raises(AttributeError):
        spectrum.power_definition = 'mean_square'
    with pytest.raises(AttributeError):
        spectrum.powers = np.ones(2)


def test_static_deformation_distinguishes_calculators_with_complex_fluctuations():
    theta = np.arange(32) * 2 * np.pi / 32
    # Rotating phases have zero complex mean but constant nonzero magnitude.
    phase = np.arange(4) * np.pi / 2
    radii = 10 + .2 * np.cos(4 * theta[None, :] + phase[:, None])
    shifted = radii + .3 * np.sin(4 * theta)
    variance = Spectrum.from_radii(radii, calculator=TemporalVarianceCalculator())
    shifted_variance = Spectrum.from_radii(shifted, calculator=TemporalVarianceCalculator())
    mean_square = Spectrum.from_radii(shifted, calculator=MeanSquaredPowerCalculator())
    q4 = np.flatnonzero(variance.modes == 4)[0]
    assert variance.powers[q4] == pytest.approx(.0001)
    assert shifted_variance.powers[q4] == pytest.approx(variance.powers[q4])
    assert mean_square.powers[q4] > shifted_variance.powers[q4]


@pytest.mark.parametrize('changes', [
    {'modes': [0., 4.]}, {'modes': [4, 4]}, {'powers': [1.]},
    {'powers': [-1., 0.]}, {'powers': [np.nan, 0.]},
    {'powers': [1j, 0j]}, {'powers': [[1., 0.]]},
    {'reference_radius': 0}, {'radius_unit': ''}, {'power_unit': ''},
    {'power_definition': ''},
])
def test_invalid_calculation_result_is_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        Spectrum.from_radii(np.ones((2, 16)), calculator=lambda radii: _result(**changes))


def test_custom_failure_is_not_suppressed_and_wrong_return_is_rejected():
    error = RuntimeError('custom calculation failed')

    def failing(radii):
        raise error

    with pytest.raises(RuntimeError) as caught:
        Spectrum.from_radii(np.ones((2, 16)), calculator=failing)
    assert caught.value is error
    with pytest.raises(TypeError, match='PowerCalculationResult'):
        Spectrum.from_radii(np.ones((2, 16)), calculator=lambda radii: np.ones(16))


def test_custom_units_and_definition_are_exported():
    spectrum = Spectrum.from_radii(
        np.ones((2, 16)),
        calculator=lambda radii: _result(reference_radius=1e-5, radius_unit='m'),
    )
    assert spectrum.r0 == pytest.approx(10)
    data = spectrum.to_dict()
    assert data['reference_radius'] == 1e-5
    assert data['radius_unit'] == 'm'
    assert data['r0'] == pytest.approx(10)
    assert data['powers'] == [1., .02]
    assert data['power_definition'] == 'custom'
    assert data['power_unit'] == 'dimensionless'
    assert 'avg_amps2' not in data
    for field in ['normalization', 'calculator_id', 'calculator_version', 'calculator_settings']:
        assert field not in data


def test_file_and_array_routes_share_calculator_and_cutoff(tmp_path):
    radii = np.vstack([np.full((2, 16), 10.), np.full((2, 16), 20.)])
    path = tmp_path / 'radii.npy'
    np.save(path, radii)
    calculator = MeanSquaredPowerCalculator()
    array = Spectrum.from_radii(radii, frame_cutoff=2, calculator=calculator)
    file = Spectrum(path, frame_cutoff=2, calculator=calculator)
    assert array.r0 == file.r0 == 10
    np.testing.assert_array_equal(array.powers, file.powers)


def test_default_matches_original_normalization_and_variance_denominator():
    radii = np.random.default_rng(7).uniform(9, 11, (3, 17))
    amps = np.fft.fft(radii, axis=1) / (radii.mean() * radii.shape[1])
    mean = Spectrum.from_radii(radii)
    variance = Spectrum.from_radii(radii, calculator=TemporalVarianceCalculator())
    np.testing.assert_allclose(mean.powers, np.mean(abs(amps)**2, axis=0))
    np.testing.assert_allclose(variance.powers, np.mean(abs(amps - amps.mean(axis=0))**2, axis=0))
