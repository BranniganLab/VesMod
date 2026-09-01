"""Unit tests for shared low-level validation helpers."""

import numpy as np
import pytest

from vesmod.validation import (
    require_finite_array,
    require_finite_real,
    require_fraction,
    require_integer,
    require_integer_valued,
    require_nonnegative_real,
    require_numeric_array,
    require_positive_real,
    require_real,
)


@pytest.mark.parametrize("value", [1, 1.5, np.int64(2), np.float64(2.5)])
def test_require_real_accepts_python_and_numpy_real_scalars(value):
    """Real Python and NumPy scalars are normalized to float."""
    result = require_real(value, "value")

    assert isinstance(result, float)
    assert result == float(value)


@pytest.mark.parametrize("value", [True, False, 1 + 2j, "1"])
def test_require_real_rejects_non_real_and_boolean_values(value):
    """Booleans and non-real values are rejected explicitly."""
    with pytest.raises(TypeError, match="value must be a real number"):
        require_real(value, "value")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_require_finite_real_rejects_nonfinite_values(value):
    """Finite-real validation rejects NaN and infinities."""
    with pytest.raises(ValueError, match="value must be finite"):
        require_finite_real(value, "value")


def test_positive_and_nonnegative_real_boundaries():
    """Positive and non-negative helpers enforce their distinct boundaries."""
    with pytest.raises(ValueError, match="positive"):
        require_positive_real(0, "value")
    assert require_nonnegative_real(0, "value") == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        require_nonnegative_real(-1, "value")


@pytest.mark.parametrize("value", [1, np.int64(2)])
def test_require_integer_accepts_integral_scalars(value):
    """Integral Python and NumPy scalars normalize to Python int."""
    result = require_integer(value, "value")

    assert isinstance(result, int)
    assert result == int(value)


@pytest.mark.parametrize("value", [True, 1.0, np.float64(2.0)])
def test_require_integer_rejects_non_integral_types(value):
    """Strict integer validation does not accept booleans or floating values."""
    with pytest.raises(TypeError, match="value must be an integer"):
        require_integer(value, "value")


def test_require_integer_valued_accepts_lmfit_style_float_values():
    """Integer-valued real numbers normalize to int while fractions fail."""
    assert require_integer_valued(3.0, "value") == 3
    assert require_integer_valued(np.float64(4.0), "value") == 4

    with pytest.raises(ValueError, match="integer-valued"):
        require_integer_valued(3.5, "value")
    with pytest.raises(ValueError, match="finite"):
        require_integer_valued(np.inf, "value")
    with pytest.raises(TypeError, match="integer-valued"):
        require_integer_valued(True, "value")


def test_require_fraction_supports_closed_and_half_open_unit_intervals():
    """Fraction validation can include or exclude the upper endpoint."""
    assert require_fraction(0, "fraction") == 0.0
    assert require_fraction(1, "fraction") == 1.0
    assert require_fraction(0.5, "fraction", include_one=False) == 0.5

    with pytest.raises(ValueError, match="less than 1"):
        require_fraction(1, "fraction", include_one=False)
    with pytest.raises(ValueError, match="at least 0"):
        require_fraction(-0.1, "fraction")


def test_array_helpers_validate_dtype_and_finiteness_without_coercion():
    """Array helpers preserve valid arrays and reject invalid array state."""
    array = np.array([[1.0, 2.0]])
    assert require_numeric_array(array, "array") is array
    assert require_finite_array(array, "array") is array

    with pytest.raises(TypeError, match="NumPy array"):
        require_numeric_array([[1.0, 2.0]], "array")
    with pytest.raises(TypeError, match="numeric values"):
        require_numeric_array(np.array([["a"]], dtype=object), "array")
    with pytest.raises(ValueError, match="finite values"):
        require_finite_array(np.array([[np.nan]]), "array")
