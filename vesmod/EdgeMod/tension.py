"""Convert between reduced and physical membrane tension."""

from scipy.constants import Boltzmann


def calc_tension_from_reduced_tension(
    r0: float,
    reduced_tension: float,
    kc: float,
    temperature: float,
) -> float:
    """
    Convert a dimensionless reduced membrane tension to a physical tension in N/m.

    The reduced tension is related to the physical membrane tension by

    sigma = tilde_sigma * kc * k_B * T / r0^2

    where ``kc`` is expressed in units of kBT and ``r0`` is the vesicle radius.

    Parameters
    ----------
    r0 : float
        Average vesicle radius in microns.
    reduced_tension : float
        Dimensionless reduced tension obtained from fitting the fluctuation
        spectrum.
    kc : float
        Membrane bending modulus in units of kBT.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Physical membrane tension in N/m (equivalently J/m²).

    Notes
    -----
    The input radius is converted from microns to meters before computing
    the tension.

    """
    one_kBT = Boltzmann * temperature                   # units of Joules
    r0_meter = r0 / 1e6                                 # units of meters
    r0_meter2 = r0_meter ** 2                           # units of meters^2
    sigma = reduced_tension * kc * one_kBT / r0_meter2  # units of J/m^2 or N/m
    return sigma
