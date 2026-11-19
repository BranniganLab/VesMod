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
from vesmod.edgemod.spectra.ensemble_member import EnsembleMember
from vesmod.edgemod.spectra.spectrum import Spectrum
from vesmod.edgemod.fitting.optimization.least_squares import fit_configured_spectrum
from vesmod.edgemod.fitting.config import SpectrumFitConfig
from vesmod.edgemod.fitting.fit_result import EnsembleFit



class SpectrumEnsemble:
    """Store and analyze fluctuation spectra from multiple replicas.

    Each replica contributes average squared mode amplitudes and supplies an
    independently determined bending modulus. All spectra must use the same Fourier modes.

    The ensemble provides averaged amplitudes, replica dispersion/error, and a
    fixed-range fit of the ensemble-averaged spectrum.

    Attributes
    ----------
    members : tuple[EnsembleMember, ...]
        Replica records in insertion order.
    modes : numpy.ndarray or None
        Fourier modes shared by all members.
    """

    def __init__(self) -> None:
        """Initialize the ensemble.

        Returns
        -------
        None
            Initializes an empty ensemble.
        """
        self._members: list[EnsembleMember] = []
        self.fit_results: list[EnsembleFit] = []

    @property
    def members(self) -> tuple[EnsembleMember, ...]:
        """Return the replica records in insertion order."""
        return tuple(self._members)

    @property
    def modes(self) -> np.ndarray | None:
        """Return the common Fourier modes, or None for an empty ensemble."""
        return self._members[0].modes if self._members else None

    def __len__(self) -> int:
        """Return the number of replica spectra.

        Returns
        -------
        int
            Number of added replica spectra.
        """
        return len(self._members)

    @property
    def avg_amps2(self) -> np.ndarray:
        """Return the mean squared fluctuation amplitudes across replicas.

        Returns
        -------
        numpy.ndarray
            Mean power per Fourier mode across replicas.
        """
        return np.mean(np.array([member.powers for member in self._members]), axis=0)

    @property
    def avg_amps2_std(self) -> np.ndarray:
        """Return the sample standard deviation of amplitudes across replicas.

        Returns
        -------
        numpy.ndarray
            Sample standard deviation per Fourier mode.
        """
        return np.std(np.array([member.powers for member in self._members]), axis=0, ddof=1)

    @property
    def avg_amps2_ste(self) -> np.ndarray:
        """Return the standard error of amplitudes across replicas.

        Returns
        -------
        numpy.ndarray
            Standard error per Fourier mode.
        """
        return self.avg_amps2_std / np.sqrt(len(self._members))

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
            Sample standard deviation of replica estimates in kBT.
        """
        return np.std([member.kC for member in self._members], ddof=1)

    @property
    def kC_ste(self) -> float:
        """Return standard error among replica kC values.

        Returns
        -------
        float
            Standard error of replica estimates in kBT.
        """
        return self.kC_std / np.sqrt(len(self._members))

    def add_spectrum(self, spectrum: Spectrum | EnsembleMember) -> None:
        """Add a measured replica spectrum.

        Parameters
        ----------
        spectrum : Spectrum or EnsembleMember
            In-memory spectrum or a member with source and fit provenance.

        Returns
        -------
        None
            Adds the member after checking compatibility.
        """
        if isinstance(spectrum, Spectrum):
            spectrum = EnsembleMember.from_spectrum(spectrum)
        if not isinstance(spectrum, EnsembleMember):
            raise TypeError("Expected a Spectrum or EnsembleMember.")
        if spectrum.power_unit != "dimensionless":
            raise ValueError("Ensemble fitting requires dimensionless powers.")
        if self._members:
            if not np.array_equal(spectrum.modes, self.modes):
                raise ValueError("Included spectra must share the same modes.")
            if spectrum.power_definition != self._members[0].power_definition:
                raise ValueError("Included spectra must share a power definition.")
        self._members.append(spectrum)

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
        config : SpectrumFitConfig or None
            Physical fit settings, or None for defaults.
        model : callable or None
            Optional factory accepting config, radii in micrometers, and expected_replica_count.
            Returns a power predictor and bounds for bending modulus and reduced tension.
        weight_by_replica_sem : bool
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
        if weight_by_replica_sem and len(self._members) < 2:
            raise ValueError(
                "At least two replica spectra are required for SEM-weighted fitting."
            )

        if not self._members:
            raise ValueError("At least one replica spectrum is required for fitting.")
        radii = tuple(member.r0 for member in self._members)
        sem = self.avg_amps2_ste if len(self._members) >= 2 else None
        result, config = fit_configured_spectrum(
            self.modes, self.avg_amps2, config, radii,
            model=model, sem=sem, weighted=weight_by_replica_sem,
            expected_replica_count=len(self._members),
        )
        kC = result.best_values["kC"]
        reduced_sigma = result.best_values["sigma"]
        fit = EnsembleFit(
            kC=float(kC),
            reduced_sigma=float(reduced_sigma),
            config=config,
            weight_by_replica_sem=weight_by_replica_sem,
            radii=radii,
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
