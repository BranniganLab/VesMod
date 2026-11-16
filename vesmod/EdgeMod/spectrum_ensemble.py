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
from .fitting import fit_configured_spectrum
from .config import SpectrumFitConfig
from .fit_result import EnsembleFit



class SpectrumEnsemble:
    """Store and analyze fluctuation spectra from multiple replicas.

    Each replica contributes average squared mode amplitudes and may supply an
    independently determined bending modulus. All spectra must use the same Fourier modes.

    The ensemble provides averaged amplitudes, replica dispersion/error, and a
    fixed-range fit of the ensemble-averaged spectrum.

    Attributes
    ----------
    spectra_list : list[np.ndarray]
        Average squared amplitudes from each replica.
    kC_list : list[float | None]
        Optional bending modulus estimate associated with each replica.
    modes : np.ndarray or None
        Fourier mode indices shared by all stored spectra. Set automatically
        when the first spectrum is added.
    """

    def __init__(self) -> None:
        """Initialize the ensemble.

        Returns
        -------
        None
            Initializes an empty ensemble.
        """
        self.spectra_list: list[np.ndarray] = []
        self.kC_list: list[float | None] = []
        self.radii_list: list[float | None] = []
        self.modes: np.ndarray[int] = None
        self.fit_results: list[EnsembleFit] = []

    def __len__(self) -> int:
        """Return the number of replica spectra.

        Returns
        -------
        int
            Number of added replica spectra.
        """
        return len(self.spectra_list)

    @property
    def avg_amps2(self) -> np.ndarray:
        """Return the mean squared fluctuation amplitudes across replicas.

        Returns
        -------
        numpy.ndarray
            Mean power per Fourier mode across replicas.
        """
        return np.mean(np.array(self.spectra_list), axis=0)

    @property
    def avg_amps2_std(self) -> np.ndarray:
        """Return the sample standard deviation of amplitudes across replicas.

        Returns
        -------
        numpy.ndarray
            Sample standard deviation per Fourier mode.
        """
        return np.std(np.array(self.spectra_list), axis=0, ddof=1)

    @property
    def avg_amps2_ste(self) -> np.ndarray:
        """Return the standard error of amplitudes across replicas.

        Returns
        -------
        numpy.ndarray
            Standard error per Fourier mode.
        """
        return self.avg_amps2_std / np.sqrt(len(self.spectra_list))

    @property
    def kC(self) -> float:
        """Return kC from a free-sigma fit of the averaged spectrum.

        Returns
        -------
        float
            Bending modulus in kBT from the default ensemble fit.
        """
        return self._extract_kC_from_fit()

    @property
    def kC_std(self) -> float:
        """Return sample standard deviation among replica kC values.

        Returns
        -------
        float
            Sample standard deviation of available replica estimates in kBT.
        """
        return np.std([kc for kc in self.kC_list if kc is not None], ddof=1)

    @property
    def kC_ste(self) -> float:
        """Return standard error among replica kC values.

        Returns
        -------
        float
            Standard error of available replica estimates in kBT.
        """
        return self.kC_std / np.sqrt(sum(kc is not None for kc in self.kC_list))

    def add_spectrum(self, avg_amps2: list[float], modes: list[int], kC: float | None = None,
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
            Replica mean radius in microns, required for every fit.
        kC : float
            Bending modulus determined independently for this replica.

        Returns
        -------
        None
            Adds the replica powers and metadata to the ensemble.

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

    def extract_kc_from_fit(
        self,
        config: SpectrumFitConfig | None = None,
        *,
        weight_by_replica_sem: bool = False,
        model=None,
    ) -> EnsembleFit:
        """Fit the averaged spectrum using a physical fit configuration.

        Reduced surface tension is fitted as a free parameter by default and
        reported as dimensionless reduced tension. Supply a configuration
        with ``free_sigma=False`` to fix it to zero. Set
        ``weight_by_replica_sem=True`` to weight each mode by the inverse
        standard error across replicas. Supply ``model`` for a custom model factory.

        Parameters
        ----------
        config
            Physical fit settings, or None for defaults.
        model
            Optional factory accepting config, radii in micrometers, and expected_replica_count.
            Returns a power predictor and bounds for bending modulus and reduced tension.
        weight_by_replica_sem
            Whether to use inverse replica SEM as weights.

        Returns
        -------
        EnsembleFit
            Fit of the mean powers, including bending modulus and reduced tension.
        """
        if config is None:
            config = SpectrumFitConfig()
        if not isinstance(config, SpectrumFitConfig):
            raise TypeError("config must be a SpectrumFitConfig or None.")
        if not isinstance(weight_by_replica_sem, bool):
            raise TypeError("weight_by_replica_sem must be a bool.")
        if weight_by_replica_sem and len(self.spectra_list) < 2:
            raise ValueError(
                "At least two replica spectra are required for SEM-weighted fitting."
            )

        if len(self.radii_list) != len(self.spectra_list) or any(
            radius is None for radius in self.radii_list
        ):
            raise ValueError("Supply r0 for every replica to set the reduced-tension bound.")
        sem = self.avg_amps2_ste if len(self.spectra_list) >= 2 else None
        result, config = fit_configured_spectrum(
            self.modes, self.avg_amps2, config, self.radii_list,
            model=model, sem=sem, weighted=weight_by_replica_sem,
            expected_replica_count=len(self.spectra_list),
        )
        kC = result.best_values["kC"]
        reduced_sigma = result.best_values["sigma"]
        fit = EnsembleFit(
            kC=float(kC),
            reduced_sigma=float(reduced_sigma),
            config=config,
            weight_by_replica_sem=weight_by_replica_sem,
            radii=tuple(self.radii_list),
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
        *,
        free_sigma: bool = True,
    ) -> float:
        """Fit a fixed q range of the averaged spectrum with free sigma by default.

        Parameters
        ----------
        lower_bound : int, default=3
            Inclusive lower Fourier mode used in the ensemble fit.
        upper_bound : int, default=8
            Exclusive upper Fourier mode used in the ensemble fit.
        lmax : int, default=500
            Maximum summation index in the theoretical spectrum model.
        free_sigma : bool, default=True
            Whether reduced surface tension is fitted as a free parameter.
            Set to False to fix it to zero.

        Returns
        -------
        float
            Best-fitting bending modulus for the averaged spectrum.
        """
        return self.extract_kc_from_fit(
            SpectrumFitConfig(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lmax=lmax,
                free_sigma=free_sigma,
            )
        ).kC
