"""Mutable synthetic spectrum state for tests of downstream consumers only."""
from vesmod.EdgeMod import Spectrum


class SyntheticSpectrum(Spectrum):
    """Bypass calculation to exercise fitting with explicitly supplied data."""

    r0 = None
    modes = None
    powers = None
    power_definition = 'mean_square'

    def to_dict(self, include_arrays=True):
        """Supply units normally provided by the calculation result."""
        from types import SimpleNamespace
        self._calculation = SimpleNamespace(power_unit='dimensionless',
                                            reference_radius=self.r0, radius_unit='um')
        return super().to_dict(include_arrays)
