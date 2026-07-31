#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:17:55 2026

@author: js2746
"""
import numpy as np
import glob
from pathlib import Path
import json
from vesmod.EdgeMod import SpectrumEnsemble

file_dir = '/home/js2746/DOPC_Cer_fluctuations/Replacement_DOPC/vesmod_test/'
sigma_max = 1.325e-7  # surface tension cutoff in N/m

ensemble = SpectrumEnsemble()

for file in glob.glob(file_dir + "*.json"):
    path = Path(file).resolve()
    check_path = path.parents[1].joinpath(path.name)
    with open(file, 'r') as json_data:
        data = json.load(json_data)

    if np.abs(data["surface_tension"]) < sigma_max:
        ensemble.add_spectrum(data["avg_amps2"], data["modes"], data["kC"])

print(ensemble.kC, ensemble.kC_ste, len(ensemble))
