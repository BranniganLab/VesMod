#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:31:11 2026

@author: js2746
"""
import numpy as np
from vesmod.EdgeMod.spectrum import MiniSpectrum
from vesmod.EdgeMod.spectrum_utils import fit_spectrum_to_theory_lmfit

class AverageOfSpectra():
    def __init__(self) -> None:
        self.spectra_list: list[np.ndarray] = []
        self.kC_list: list[float] = []
        self.modes: np.ndarray[int] = None

    @property
    def avg_amps2(self) -> np.ndarray:
        return np.mean(np.array(self.spectra_list), axis=0)

    @property
    def kC(self) -> float:
        return self._extract_kC_from_fit()

    @property
    def kC_std(self) -> float:
        return np.std(self.kC_list, ddof=1)

    @property
    def kC_ste(self) -> float:
        return self.kC_std / np.sqrt(len(self.kC_list))

    def add_spectrum(self, avg_amps2: np.ndarray, modes: np.ndarray[int], kC: float) -> None:
        if isinstance(self.modes, np.ndarray):
            if not np.array_equal(modes, self.modes):
                raise ValueError(f"{modes} does not equal {self.modes}")
        elif self.modes is None:
            self.modes = modes
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

    def _extract_kc_from_fit(
        self,
        lower_bound: int = 3,
        upper_bound: int = 8,
        lmax: int = 500,
    ) -> tuple(float, float):
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
