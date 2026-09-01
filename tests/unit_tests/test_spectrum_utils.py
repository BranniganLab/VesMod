#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/test_spectrum_utils.py
import math

import numpy as np
import pytest
from scipy.constants import Boltzmann
from scipy.special import lpmv

from vesmod.EdgeMod.spectrum_utils import (
    HSS97,
    MiniSpectrum,
    Nlq_Plq0_squared,
    calc_tension_from_reduced_tension,
    fit_spectrum_to_theory_lmfit,
)


def test_mini_spectrum_stores_modes_avg_amps2_and_std_amps2():
    """Test that MiniSpectrum stores exactly the modes, average amplitudes, and standard deviations provided."""
    modes = np.array([3, 4])
    avg_amps2 = np.array([0.1, 0.2])
    std_amps2 = np.array([0.01, 0.02])

    spectrum = MiniSpectrum(modes, avg_amps2, std_amps2)

    assert spectrum.modes is modes
    assert spectrum.avg_amps2 is avg_amps2
    assert spectrum.std_amps2 is std_amps2


def test_calc_tension_from_reduced_tension_converts_microns_to_meters():
    """Test that reduced tension is converted to physical tension using r0 in microns and kBT in Joules."""
    r0 = 10.0
    reduced_tension = 2.0
    kc = 20.0
    temperature = 295.0

    expected = reduced_tension * kc * Boltzmann * temperature / (r0 / 1e6) ** 2

    assert calc_tension_from_reduced_tension(r0, reduced_tension, kc, temperature) == pytest.approx(expected)


def test_calc_tension_from_reduced_tension_returns_zero_for_zero_reduced_tension():
    """Test that physical tension is exactly zero when the fitted reduced tension is zero."""
    assert calc_tension_from_reduced_tension(10.0, 0.0, 20.0, 295.0) == 0.0


def test_nlq_plq0_squared_matches_known_l2_q0_value():
    """Test N_lq P_lq(0)^2 for l=2 and q=0, where P_20(0)=-1/2 and the expected value is 5/(16*pi)."""
    assert Nlq_Plq0_squared(2, 0) == pytest.approx(5 / (16 * math.pi))


@pytest.mark.parametrize(
    ("l", "q"),
    [(2, 0), (2, 2), (3, 1), (6, 4), (10, 3)],
)
def test_nlq_plq0_squared_matches_direct_low_mode_calculation(l, q):
    """Test the stable expression retains established low-mode values."""
    normalization = (2 * l + 1) / (4 * np.pi)
    normalization *= math.factorial(l - q) / math.factorial(l + q)
    expected = normalization * lpmv(q, l, 0) ** 2

    assert Nlq_Plq0_squared(l, q) == pytest.approx(expected)


def test_nlq_plq0_squared_returns_zero_for_odd_parity():
    """Test P_lq(0) is exactly zero when l + q is odd."""
    assert Nlq_Plq0_squared(l=89, q=88) == 0.0


def test_nlq_plq0_squared_is_finite_at_l89_q89():
    """Test the high-mode pair from the reported failure remains finite."""
    result = Nlq_Plq0_squared(l=89, q=89)

    assert np.isfinite(result)
    assert result > 0.0


def test_hss97_succeeds_for_q89():
    """Test the full theoretical sum supports a high diagnostic mode."""
    result = HSS97(q=[89], kC=20.0, sigma=0.0, lmax=500)

    assert len(result) == 1
    assert np.isfinite(result[0])
    assert result[0] > 0.0


def test_nlq_plq0_squared_requires_q_not_greater_than_l():
    """Test that Nlq_Plq0_squared rejects q > l explicitly."""
    with pytest.raises(ValueError, match="q must be <= l"):
        Nlq_Plq0_squared(l=2, q=3)


def test_nlq_plq0_squared_requires_nonnegative_q():
    """Test that Nlq_Plq0_squared rejects negative q explicitly."""
    with pytest.raises(ValueError, match="q must be non-negative"):
        Nlq_Plq0_squared(l=2, q=-1)


def test_nlq_plq0_squared_does_not_change_numpy_error_policy():
    """Test local floating-point checks do not affect later calculations."""
    original_settings = np.geterr()

    Nlq_Plq0_squared(l=2, q=0)

    assert np.geterr() == original_settings


def test_hss97_returns_one_value_per_input_mode():
    """Test that HSS97 returns one theoretical spectrum value for each requested Fourier mode."""
    q = np.array([2, 3, 4])

    values = HSS97(q=q, kC=20.0, sigma=0.0, lmax=20)

    assert isinstance(values, list)
    assert len(values) == len(q)


def test_hss97_uses_range_from_q_to_lmax_exclusive():
    """Test that HSS97 sums over l from q through lmax-1, matching Python's range(q, lmax) behavior."""
    q = 2
    kC = 5.0
    sigma = 0.0
    lmax = 4

    expected_sum = 0.0
    for l in range(q, lmax):
        denom = (l - 1) * (l + 2) * (l**2 + l + sigma)
        expected_sum += Nlq_Plq0_squared(l, q) / denom

    assert HSS97(q=[q], kC=kC, sigma=sigma, lmax=lmax)[0] == pytest.approx(expected_sum / kC)


def test_fit_spectrum_to_theory_lmfit_recovers_synthetic_kc_when_sigma_is_fixed():
    """Test that fitting synthetic HSS97 data with fixed sigma recovers the known kC used to generate the data."""
    modes = np.array([2, 3, 4, 5, 6])
    true_kc = 25.0
    lmax = 60
    avg_amps2 = np.array(HSS97(modes, kC=true_kc, sigma=0.0, lmax=lmax))
    fitting_group = MiniSpectrum(modes, avg_amps2, None)

    kc, sigma = fit_spectrum_to_theory_lmfit(fitting_group, lmax=lmax, free_sigma=False)

    assert kc == pytest.approx(true_kc, rel=1e-3)
    assert sigma == pytest.approx(0.0, abs=1e-10)
