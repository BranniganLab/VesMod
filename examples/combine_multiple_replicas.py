#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:17:55 2026

@author: js2746
"""

import glob
from pathlib import Path
import json
from vesmod.EdgeMod import Spectrum, AverageOfSpectra

file_dir = 'YOUR/PATH/HERE/'
sigma_max = 1.325e-7  # surface tension cutoff in N/m


for file in glob(file_dir + "*.json"):
    path = Path(file).resolve()
    with open(file, 'r') as json_data:
        data = json.load(json_data)
    