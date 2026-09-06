"""Experimental EdgeMod analysis features.

APIs in this package are intentionally not part of the stable core EdgeMod
interface and may change as the underlying methods are evaluated.
"""

from .dynamic_range import DynamicRangeSelection, QMinusThreeRangeSelector

__all__ = [
    "DynamicRangeSelection",
    "QMinusThreeRangeSelector",
]
