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
from vesmod.EdgeMod import AverageOfSpectra

file_dir = '/home/js2746/DOPC_Cer_fluctuations/Replacement_DOPC/vesmod_test/'
sigma_max = 1.325e-7  # surface tension cutoff in N/m

average = AverageOfSpectra()

for file in glob.glob(file_dir + "*.json"):
    path = Path(file).resolve()
    check_path = path.parents[1].joinpath(path.name)
    with open(file, 'r') as json_data:
        data = json.load(json_data)
    with open(check_path, 'r') as check_data:
        test = json.load(check_data)

    if not np.allclose(data['kC'], test['kC_3_8'][0]):
        print(f"{path.name}")
    # if not np.allclose(npdata, checknpdata):
        # print(f"{path.name} npy files differ")
    checkr0 = test["r0"]
    filt = len(data["filtered_data"])
    filt2 = test["frame_count"][1]
    print(path.stem)
    print(filt, filt2)
    # if r0 != checkr0:
        # print(path.stem)
        # print(r0-checkr0)
    # if not np.allclose(data['kC'], test["kC_3_8"][0]):
        # print(f"{path.name}")
        # print(data['kC'] - test["kC_3_8"][0])
    # if np.abs(data["surface_tension"]) < sigma_max:
        # average.add_spectrum(data["avg_amps2"], data["modes"], data["kC"])

# print(average.kC, average.kC_ste)
# print(np.mean(average.kC_list))