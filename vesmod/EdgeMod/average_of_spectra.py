#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:31:11 2026

@author: js2746
"""
import numpy as np
from vesmod.EdgeMod import Spectrum

class AverageOfSpectra(Spectrum):
    def __init__(self) -> None:
        self.spectra_list: list[Spectrum] = []
        self.kC_list: list[float] = []
        self.modes: np.ndarray[int] = None

    @property
    def avg_amps2(self) -> np.ndarray:
        return np.mean(np.array(self.spectra_list), axis=0)

    @property
    def kC(self) -> float:
        return self.extract_kC_from_fit(free_sigma=False)[0]

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
