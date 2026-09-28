"""Experimental EdgeMod analysis features.

APIs in this package are intentionally not part of the stable core EdgeMod
interface and may change as the underlying methods are evaluated.
"""

from .dynamic_range import DynamicRangeSelection, QMinusThreeRangeSelector
from .mode_rigidity import (
    ModeRigidityResult,
    calculate_mode_rigidity,
    plot_mode_rigidity,
)

__all__ = [
    "DynamicRangeSelection",
    "QMinusThreeRangeSelector",
    "ModeRigidityResult",
    "calculate_mode_rigidity",
    "plot_mode_rigidity",
]
