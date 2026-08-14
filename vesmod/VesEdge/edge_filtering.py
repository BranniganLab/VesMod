#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:49:23 2026

@author: js2746
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class QCFlag(Enum):
    CURVATURE = auto()
    IMAGE_SUPPORT = auto()
    POPULATION_OUTLIER = auto()
    ERROR = auto()


@dataclass
class EdgeQC:
    flags: set[QCFlag] = field(default_factory=set)

    curvature_score: float | None = None
    image_support_fraction: float | None = None
    population_probability: float | None = None

    @property
    def passed(self) -> bool:
        return not self.flags