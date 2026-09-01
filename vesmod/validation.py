"""Shared low-level validation helpers for VesMod.

This module centralizes generic validation mechanics only. Scientific and
workflow-specific policy belongs in the module that owns the corresponding
configuration, domain object, or operation.
"""

from numbers import Integral, Real

import numpy as np


def require_real(
    value,
    name: str,
    *,
    type_message: str | None = None,
) -> float:
    """Return ``value`` as a float after requiring a non-boolean real number.

    Parameters
    ----------
    value : object
        Value to validate.
    name : str
        Parameter name used in validation errors.
    type_message : str | None, default=None
        Optional replacement for the default type-error message.

    Returns
    -------
    float
        Validated value converted to ``float``.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(type_message or f"{name} must be a real number.")
    return float(value)


def require_finite_real(
    value,
    name: str,
    *,
    type_message: str | None = None,
    finite_message: str | None = None,
) -> float:
    """Return a finite non-boolean real number as ``float``."""
    validated = require_real(value, name, type_message=type_message)
    if not np.isfinite(validated):
        raise ValueError(finite_message or f"{name} must be finite.")
    return validated


def require_positive_real(
    value,
    name: str,
    *,
    type_message: str | None = None,
    finite_message: str | None = None,
    range_message: str | None = None,
) -> float:
    """Return a finite real number greater than zero."""
    validated = require_finite_real(
        value,
        name,
        type_message=type_message,
        finite_message=finite_message,
    )
    if validated <= 0:
        raise ValueError(range_message or f"{name} must be positive.")
    return validated


def require_nonnegative_real(
    value,
    name: str,
    *,
    type_message: str | None = None,
    finite_message: str | None = None,
    range_message: str | None = None,
) -> float:
    """Return a finite real number greater than or equal to zero."""
    validated = require_finite_real(
        value,
        name,
        type_message=type_message,
        finite_message=finite_message,
    )
    if validated < 0:
        raise ValueError(range_message or f"{name} must be non-negative.")
    return validated


def require_integer(
    value,
    name: str,
    *,
    type_message: str | None = None,
) -> int:
    """Return a non-boolean integer value as a Python ``int``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(type_message or f"{name} must be an integer.")
    return int(value)


def require_integer_valued(
    value,
    name: str,
    *,
    type_message: str | None = None,
    finite_message: str | None = None,
    integer_message: str | None = None,
) -> int:
    """Return a finite integer-valued real number as a Python ``int``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            type_message or f"{name} must be an integer-valued number."
        )
    if not np.isfinite(value):
        raise ValueError(finite_message or f"{name} must be finite.")
    if not float(value).is_integer():
        raise ValueError(integer_message or f"{name} must be integer-valued.")
    return int(value)


def require_fraction(
    value,
    name: str,
    *,
    include_one: bool = True,
    type_message: str | None = None,
    finite_message: str | None = None,
    range_message: str | None = None,
) -> float:
    """Return a finite fraction bounded between zero and one.

    Parameters
    ----------
    value : object
        Value to validate.
    name : str
        Parameter name used in validation errors.
    include_one : bool, default=True
        Whether exactly one is an allowed upper-bound value.
    type_message : str | None, default=None
        Optional replacement for the default type-error message.
    finite_message : str | None, default=None
        Optional replacement for the default non-finite-value message.
    range_message : str | None, default=None
        Optional replacement for the default range-error message.

    Returns
    -------
    float
        Validated fraction.
    """
    validated = require_finite_real(
        value,
        name,
        type_message=type_message,
        finite_message=finite_message,
    )
    upper_valid = validated <= 1 if include_one else validated < 1
    if validated < 0 or not upper_valid:
        relation = "at most 1" if include_one else "less than 1"
        raise ValueError(
            range_message or f"{name} must be at least 0 and {relation}."
        )
    return validated


def require_numeric_array(value, name: str) -> np.ndarray:
    """Return a NumPy array after requiring numeric dtype.

    The helper intentionally requires an existing ``numpy.ndarray`` rather
    than coercing arbitrary array-like inputs. Callers that intentionally
    support array-like values should perform that coercion at their own API
    boundary before using this helper.
    """
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values.")
    return value


def require_finite_array(value: np.ndarray, name: str) -> np.ndarray:
    """Return an array after requiring every element to be finite."""
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")
    return value
