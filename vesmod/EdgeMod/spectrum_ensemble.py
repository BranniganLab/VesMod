#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for averaging fluctuation spectra across multiple replicas.

This module provides the SpectrumEnsemble class, which aggregates
Spectrum objects and bending modulus estimates from multiple replicas
of the same system. The class calculates ensemble-averaged squared
amplitudes, replica-to-replica uncertainty estimates, and an
overall bending modulus obtained by fitting the averaged spectrum
to the theoretical fluctuation model.

Typical usage consists of adding one Spectrum per replica using
:add_spectrum:, then accessing the averaged amplitudes through
:attr:`avg_amps2` or the fitted bending modulus through
:attr:`kC`.
"""
import numpy as np
from vesmod.EdgeMod.spectrum_utils import fit_spectrum_to_theory_lmfit, MiniSpectrum


class SpectrumEnsemble:
    """
    Store and analyze fluctuation spectra from multiple replicas.

    Each replica contributes a spectrum of average squared mode
    amplitudes and an independently determined bending modulus.
    All spectra must be defined on the same set of Fourier modes.

    The class provides properties for calculating:

    * Ensemble-averaged squared amplitudes.
    * Standard deviations and standard errors across replicas.
    * Mean bending modulus obtained by fitting the averaged spectrum.
    * Uncertainty estimates based on replica-to-replica variation.

    Attributes
    ----------
    spectra_list : list[np.ndarray]
        Average squared amplitudes from each replica.
    kC_list : list[float]
        Bending modulus estimate associated with each replica.
    modes : np.ndarray or None
        Fourier mode indices shared by all stored spectra.
        Set automatically when the first spectrum is added.
    """

    def __init__(self) -> None:
        self.spectra_list: list[np.ndarray] = []
        self.kC_list: list[float] = []
        self.modes: np.ndarray[int] = None

    def __len__(self) -> int:
        """Return length of kC_list."""
        return len(self.kC_list)

    @property
    def avg_amps2(self) -> np.ndarray:
        """Return mean of spectra_list along time dimension."""
        return np.mean(np.array(self.spectra_list), axis=0)

    @property
    def avg_amps2_std(self) -> np.ndarray:
        """Take standard deviation of spectra_list along time dimension."""
        return np.std(np.array(self.spectra_list), axis=0, ddof=1)

    @property
    def avg_amps2_ste(self) -> np.ndarray:
        """Calculate standard error."""
        return self.avg_amps2_std / np.sqrt(len(self.spectra_list))

    @property
    def kC(self) -> float:
        """Calculate kC from fit."""
        return self._extract_kC_from_fit()

    @property
    def kC_std(self) -> float:
        """Calculate standard deviation from dispersion among replica kC values."""
        return np.std(self.kC_list, ddof=1)

    @property
    def kC_ste(self) -> float:
        """Calculate standard error."""
        return self.kC_std / np.sqrt(len(self.kC_list))

    def add_spectrum(self, avg_amps2: list[float], modes: list[int], kC: float) -> None:
        """
        Add a spectrum replica to the ensemble.

        Parameters
        ----------
        avg_amps2 : array-like of float
            Average squared fluctuation amplitudes for each Fourier mode.
        modes : array-like of int
            Fourier mode indices corresponding to ``avg_amps2``.
            The mode array must match that of all previously added
            spectra.
        kC : float
            Bending modulus determined from this replica.

        Raises
        ------
        ValueError
            If ``modes`` does not match the mode indices already stored
            in the object.
        TypeError
            If ``self.modes`` is neither ``None`` nor a NumPy array.

        Notes
        -----
        The first spectrum added defines the mode indices used by the
        ensemble. All subsequent spectra must use the same modes so that
        replica averages can be computed element-wise.

        """
        if isinstance(self.modes, np.ndarray):
            if not np.array_equal(np.array(modes), self.modes):
                raise ValueError(f"{modes} does not equal {self.modes}")
        elif self.modes is None:
            self.modes = np.array(modes)
        else:
            raise TypeError(f"self.modes must be ndarray or None, not {type(self.modes)}")
        self.spectra_list.append(avg_amps2)
        self.kC_list.append(kC)

    def _isolate_mode_range(self, lower_bound: int, upper_bound: int) -> MiniSpectrum:
        """
        Return all modes greater than or equal to lower_bound and less than \
        upper_bound, and their associated avg squared amplitudes.

        Returns
        -------
        MiniSpectrum : namedtuple

        """
        if self.modes is None:
            raise AttributeError("There are no modes; Cannot return mode range.")
        mask1 = self.modes >= lower_bound
        mask2 = self.modes < upper_bound
        combined_mask = mask1 & mask2
        return MiniSpectrum(self.modes[combined_mask], self.avg_amps2[combined_mask], None)

    def _extract_kC_from_fit(
        self,
        lower_bound: int = 3,
        upper_bound: int = 8,
        lmax: int = 500,
    ) -> tuple[float, float]:
        """
        Fit specific range of self.avg_amps2 to theoretical prediction.

        Parameters
        ----------
        lower_bound, upper_bound : ints
            Fit modes greater than or equal to lower_bound and less than upper_bound.
        lmax : int
            Maximum iteration index in theoretical summation. Default is 500.
        free_sigma : bool
            If True, allow surface tension (sigma) to vary. If False, set sigma
            to zero and do not let it vary during fitting.

        Returns
        -------
        float
            The best fitting kC within the fitting range with sigma set to 0.

        """
        fitting_range = self._isolate_mode_range(lower_bound, upper_bound)
        fit = fit_spectrum_to_theory_lmfit(fitting_range, lmax, free_sigma=False)
        return fit[0]
