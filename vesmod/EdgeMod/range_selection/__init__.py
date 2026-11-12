"""Select the measured Fourier modes to use in a spectrum fit."""


def select_manual_range(modes, lower_bound, upper_bound):
    """Return a Boolean array selecting modes in the user-supplied q interval."""
    if modes is None:
        raise AttributeError("There are no modes; Cannot return mode range.")
    return (modes >= lower_bound) & (modes < upper_bound)


def slice_fit_arrays(fit_mode_mask, modes, powers, *, sem=None):
    """Apply one mode selection to powers and optional per-mode SEM."""
    return (
        modes[fit_mode_mask],
        powers[fit_mode_mask],
        None if sem is None else sem[fit_mode_mask],
    )
