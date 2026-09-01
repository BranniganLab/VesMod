"""Regression tests for EdgeMod input and HSS97 domain validation."""

import numpy as np
import pytest

from vesmod.EdgeMod import Spectrum, SpectrumFitConfig
from vesmod.EdgeMod.spectrum_utils import HSS97, Nlq_Plq0_squared


@pytest.mark.parametrize(
    "radii, error_type, message",
    [
        (np.ones(4), ValueError, "two-dimensional"),
        (np.empty((0, 4)), ValueError, "at least one frame"),
        (np.ones((2, 1)), ValueError, "at least two angular samples"),
        (np.array([["a", "b"]], dtype=object), TypeError, "numeric"),
        (np.array([[1.0, np.nan]]), ValueError, "finite"),
        (np.array([[1.0, np.inf]]), ValueError, "finite"),
        (np.array([[1.0, 0.0]]), ValueError, "positive"),
        (np.array([[1.0, -1.0]]), ValueError, "positive"),
    ],
)
def test_from_radii_rejects_invalid_arrays(radii, error_type, message):
    """Invalid contour arrays fail before FFT calculation."""
    with pytest.raises(error_type, match=message):
        Spectrum.from_radii(radii)


def test_from_radii_constructs_spectrum_and_copies_input():
    """In-memory construction matches file construction semantics and owns data."""
    radii = np.array([[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]])

    spectrum = Spectrum.from_radii(radii)
    radii[:] = 99.0

    assert spectrum.r0 == pytest.approx(2.5)
    np.testing.assert_allclose(
        spectrum.avg_amps2,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_file_constructor_uses_same_radii_validation(tmp_path):
    """File-loaded arrays pass through the common validation path."""
    path = tmp_path / "bad.npy"
    np.save(path, np.array([[1.0, np.nan]]))

    with pytest.raises(ValueError, match="finite"):
        Spectrum(path)


def test_fit_config_rejects_q1_domain():
    """The HSS97 singular q=1 mode is rejected before fitting."""
    with pytest.raises(ValueError, match="at least 2"):
        SpectrumFitConfig(lower_bound=1, upper_bound=4)


def test_fit_config_requires_lmax_to_cover_fit_interval():
    """lmax must include an l=q term for the highest configured q mode."""
    with pytest.raises(ValueError, match="at least upper_bound"):
        SpectrumFitConfig(lower_bound=3, upper_bound=8, lmax=7)

    config = SpectrumFitConfig(lower_bound=3, upper_bound=8, lmax=8)
    assert config.lmax == 8


def test_fit_config_requires_enough_modes_for_free_parameters():
    """A two-parameter fit cannot be configured with only one q mode."""
    with pytest.raises(ValueError, match="too few modes"):
        SpectrumFitConfig(lower_bound=3, upper_bound=4, free_sigma=True)

    config = SpectrumFitConfig(lower_bound=3, upper_bound=4, free_sigma=False)
    assert config.lower_bound == 3
    assert config.upper_bound == 4


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"q": [1], "kC": 20.0, "sigma": 0.0, "lmax": 10}, ValueError, "q >= 2"),
        ({"q": [10], "kC": 20.0, "sigma": 0.0, "lmax": 10}, ValueError, "less than lmax"),
        ({"q": [3], "kC": 0.0, "sigma": 0.0, "lmax": 10}, ValueError, "positive"),
        ({"q": [3], "kC": 20.0, "sigma": np.nan, "lmax": 10}, ValueError, "finite"),
        ({"q": [3], "kC": 20.0, "sigma": 0.0, "lmax": 3.5}, ValueError, "integer-valued"),
    ],
)
def test_hss97_rejects_invalid_public_domain(kwargs, error_type, message):
    """Direct model calls fail with explicit domain errors."""
    with pytest.raises(error_type, match=message):
        HSS97(**kwargs)


def test_hss97_accepts_lmfit_style_float_lmax():
    """A fixed lmfit parameter may reach HSS97 as an integral-valued float."""
    result = HSS97(q=np.array([3, 4]), kC=20.0, sigma=0.0, lmax=10.0)
    assert len(result) == 2
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "l, q, error_type, message",
    [
        (2, 3, ValueError, "q must be <= l"),
        (2, -1, ValueError, "non-negative"),
        (2.5, 2, TypeError, "l must be an integer"),
    ],
)
def test_nlq_validation_uses_explicit_exceptions(l, q, error_type, message):
    """Nlq argument validation does not rely on assertions."""
    with pytest.raises(error_type, match=message):
        Nlq_Plq0_squared(l=l, q=q)
