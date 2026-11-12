"""Select the measured Fourier modes to use in a spectrum fit."""

from collections import namedtuple

MiniSpectrum = namedtuple("MiniSpectrum", ["modes", "avg_amps2", "avg_amps2_ste"])


def select_manual_range(modes, powers, lower_bound, upper_bound, *, sem=None):
    """Slice measured powers and optional SEM to the user-supplied q interval."""
    if modes is None:
        raise AttributeError("There are no modes; Cannot return mode range.")
    mask = (modes >= lower_bound) & (modes < upper_bound)
    return MiniSpectrum(modes[mask], powers[mask], None if sem is None else sem[mask])
