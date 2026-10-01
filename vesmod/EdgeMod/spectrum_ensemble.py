#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tools for averaging fluctuation spectra across multiple replicas.

``SpectrumEnsemble`` aggregates spectra and bending-modulus estimates from
replicas of the same system. It computes ensemble-averaged squared amplitudes,
replica-to-replica uncertainty estimates, and a bending modulus from a fixed
q-range fit of the averaged spectrum.

Dynamic q-range selection introduced for ``Spectrum`` is not applied
implicitly to ensemble fits; ensemble fitting retains its existing fixed-range
behavior.
"""
import numpy as np
from vesmod.validation import require_positive_real
from vesmod.EdgeMod.spectrum_utils import (
    fit_spectrum_lmfit,
    MiniSpectrum,
    validate_lmfit_result,
)
from .config import SpectrumFitConfig
from .fit_result import EnsembleFit


class SpectrumEnsemble:
    """Store and analyze fluctuation spectra from multiple replicas.

    Each replica contributes average squared mode amplitudes and an independently
    determined bending modulus. All spectra must use the same Fourier modes.

    The ensemble provides averaged amplitudes, replica dispersion/error, and a
    fixed-range fit of the ensemble-averaged spectrum.

    Attributes
    ----------
    spectra_list : list[np.ndarray]
        Average squared amplitudes from each replica.
    kC_list : list[float]
        Bending modulus estimate associated with each replica.
    modes : np.ndarray or None
        Fourier mode indices shared by all stored spectra. Set automatically
        when the first spectrum is added.
    """

    def __init__(self) -> None:
        self.spectra_list: list[np.ndarray] = []
        self.kC_list: list[float] = []
        self.radii_list: list[float | None] = []
        self.modes: np.ndarray[int] = None
        self.fit_results: list[EnsembleFit] = []

    def __len__(self) -> int:
        """Return the number of replica bending-modulus estimates."""
        return len(self.kC_list)

    @property
    def avg_amps2(self) -> np.ndarray:
        """Return the mean squared fluctuation amplitudes across replicas."""
        return np.mean(np.array(self.spectra_list), axis=0)

    @property
    def avg_amps2_std(self) -> np.ndarray:
        """Return the sample standard deviation of amplitudes across replicas."""
        return np.std(np.array(self.spectra_list), axis=0, ddof=1)

    @property
    def avg_amps2_ste(self) -> np.ndarray:
        """Return the standard error of amplitudes across replicas."""
        return self.avg_amps2_std / np.sqrt(len(self.spectra_list))

    @property
    def kC(self) -> float:
        """Return kC from the legacy fixed-sigma fit of the averaged spectrum."""
        return self._extract_kC_from_fit()

    @property
    def kC_std(self) -> float:
        """Return sample standard deviation among replica kC values."""
        return np.std(self.kC_list, ddof=1)

    @property
    def kC_ste(self) -> float:
        """Return standard error among replica kC values."""
        return self.kC_std / np.sqrt(len(self.kC_list))

    def add_spectrum(self, avg_amps2: list[float], modes: list[int], kC: float,
                     *, r0: float | None = None) -> None:
        """Add one replica spectrum and its fitted bending modulus.

        Parameters
        ----------
        avg_amps2 : array-like of float
            Average squared fluctuation amplitudes for each Fourier mode.
        modes : array-like of int
            Fourier mode indices corresponding to ``avg_amps2``. The mode array
            must match all previously added spectra.
        r0 : float or None
            Replica mean radius in microns, required for exposure-aware fits.
        kC : float
            Bending modulus determined independently for this replica.

        Raises
        ------
        ValueError
            If ``modes`` does not match the mode indices already stored.
        TypeError
            If ``self.modes`` is neither ``None`` nor a NumPy array.
        """
        if r0 is not None:
            r0 = require_positive_real(r0, "r0")
        if isinstance(self.modes, np.ndarray):
            if not np.array_equal(np.array(modes), self.modes):
                raise ValueError(f"{modes} does not equal {self.modes}")
        elif self.modes is None:
            self.modes = np.array(modes)
        else:
            raise TypeError(f"self.modes must be ndarray or None, not {type(self.modes)}")
        self.spectra_list.append(avg_amps2)
        self.kC_list.append(kC)
        self.radii_list.append(r0)

    def _isolate_mode_range(self, lower_bound: int, upper_bound: int) -> MiniSpectrum:
        """Return ensemble modes with ``lower_bound <= q < upper_bound``."""
        if self.modes is None:
            raise AttributeError("There are no modes; Cannot return mode range.")
        mask1 = self.modes >= lower_bound
        mask2 = self.modes < upper_bound
        combined_mask = mask1 & mask2
        avg_amps2_ste = (
            self.avg_amps2_ste[combined_mask]
            if len(self.spectra_list) >= 2
            else None
        )
        return MiniSpectrum(
            self.modes[combined_mask],
            self.avg_amps2[combined_mask],
            avg_amps2_ste,
        )

    def extract_kc_from_fit(
        self,
        config: SpectrumFitConfig | None = None,
        *,
        weight_by_replica_sem: bool = False,
    ) -> EnsembleFit:
        """Fit the averaged spectrum using a physical fit configuration.

        A free-sigma ensemble fit reports HSS97 reduced surface tension. When
        no configuration is supplied, the historical fixed-sigma behavior is
        retained. Set ``weight_by_replica_sem=True`` to weight each mode by the
        inverse standard error across replicas.
        """
        if config is None:
            config = SpectrumFitConfig(free_sigma=False)
        if not isinstance(config, SpectrumFitConfig):
            raise TypeError("config must be a SpectrumFitConfig or None.")
        if not isinstance(weight_by_replica_sem, bool):
            raise TypeError("weight_by_replica_sem must be a bool.")
        if weight_by_replica_sem and len(self.spectra_list) < 2:
            raise ValueError(
                "At least two replica spectra are required for SEM-weighted fitting."
            )

        camera_kwargs = {}
        if config.exposure_time > 0:
            if len(self.radii_list) != len(self.spectra_list) or any(
                radius is None for radius in self.radii_list
            ):
                raise ValueError("Supply r0 for every replica to use camera integration.")
            camera_kwargs = {"config": config, "radii": self.radii_list}
        fitting_range = self._isolate_mode_range(
            config.lower_bound,
            config.upper_bound,
        )
        result = fit_spectrum_lmfit(
            fitting_range,
            config.lmax,
            free_sigma=config.free_sigma,
            weighted=weight_by_replica_sem,
            **camera_kwargs,
        )
        validate_lmfit_result(result, fitting_range, config.free_sigma)
        kC = result.best_values["kC"]
        reduced_sigma = result.best_values["sigma"]
        fit = EnsembleFit(
            kC=float(kC),
            reduced_sigma=float(reduced_sigma),
            config=config,
            weight_by_replica_sem=weight_by_replica_sem,
            radii=tuple(self.radii_list) if config.exposure_time > 0 else None,
            chisqr=float(result.chisqr),
            redchi=(float(result.redchi) if getattr(result, "nfree", 1) > 0 else None),
        )
        self.fit_results.append(fit)
        return fit

    def _extract_kC_from_fit(
        self,
        lower_bound: int = 3,
        upper_bound: int = 8,
        lmax: int = 500,
    ) -> float:
        """Fit a fixed q range of the averaged spectrum with sigma fixed to zero.

        Parameters
        ----------
        lower_bound : int, default=3
            Inclusive lower Fourier mode used in the ensemble fit.
        upper_bound : int, default=8
            Exclusive upper Fourier mode used in the ensemble fit.
        lmax : int, default=500
            Maximum summation index in the theoretical spectrum model.

        Returns
        -------
        float
            Best-fitting bending modulus for the averaged spectrum with reduced
            surface tension fixed to zero.
        """
        return self.extract_kc_from_fit(
            SpectrumFitConfig(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lmax=lmax,
                free_sigma=False,
            )
        ).kC
