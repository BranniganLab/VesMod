"""Standard results returned by every VesEdge quality-control check."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Integral, Real

import numpy as np


def _json_value(value):
    """Normalize measurements to JSON data, using null for nonfinite numbers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("QC measurement keys must be strings.")
        return {key: _json_value(item) for key, item in value.items()}
    raise TypeError(f"Unsupported QC measurement type: {type(value).__name__}.")


@dataclass(frozen=True)
class QCResult:
    """One check's decisions and measurements in source-frame order.

    Each diagnostic column has one value per frame. Recording-level reasons
    apply to the whole trajectory. Additional aggregate measurements belong
    in extras and must be declared by the check's extra_result_fields.
    Nonfinite measurements are represented by None.
    """

    check_name: str
    frame_indices: tuple[int, ...]
    rejected: tuple[bool, ...]
    diagnostics: Mapping[str, tuple[object, ...]] = field(default_factory=dict)
    trajectory_flags: frozenset[str] = frozenset()
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.check_name, str) or not self.check_name:
            raise ValueError("QCResult.check_name must be a nonempty string.")
        indices = tuple(self.frame_indices)
        if any(isinstance(index, bool) or not isinstance(index, Integral) or index < 0
               for index in indices):
            raise ValueError("QC frame indices must be nonnegative integers.")
        if any(left >= right for left, right in zip(indices, indices[1:])):
            raise ValueError("QC frame indices must be unique and increasing.")
        rejected = tuple(self.rejected)
        if len(rejected) != len(indices) or any(type(value) is not bool for value in rejected):
            raise ValueError("QC rejection decisions must contain one boolean per frame.")
        diagnostics = _json_value(self.diagnostics)
        extras = _json_value(self.extras)
        if not isinstance(diagnostics, dict) or not isinstance(extras, dict):
            raise TypeError("QC diagnostics and extras must be mappings.")
        for name, values in diagnostics.items():
            if not isinstance(values, list) or len(values) != len(indices):
                raise ValueError(f"QC diagnostic {name!r} must contain one value per frame.")
        if not isinstance(self.trajectory_flags, (tuple, list, set, frozenset)):
            raise TypeError("QC trajectory flags must be a collection of strings.")
        if any(not isinstance(reason, str) or not reason for reason in self.trajectory_flags):
            raise ValueError("QC trajectory flags must be nonempty strings.")
        object.__setattr__(self, "frame_indices", tuple(int(index) for index in indices))
        object.__setattr__(self, "rejected", rejected)
        object.__setattr__(self, "diagnostics", {
            name: tuple(values) for name, values in diagnostics.items()
        })
        object.__setattr__(self, "extras", extras)
        object.__setattr__(self, "trajectory_flags", frozenset(self.trajectory_flags))

    @property
    def rejected_count(self) -> int:
        """Return the number of individually rejected frames."""
        return sum(self.rejected)

    @classmethod
    def from_detections(
        cls, check_name, detections, diagnostic_fields=(), *, extras=None,
    ) -> QCResult:
        """Collect a check's decisions and declared diagnostics from detections."""
        return cls(
            check_name=check_name,
            frame_indices=tuple(edge.frame_index for edge in detections),
            rejected=tuple(check_name in edge.qc.flags for edge in detections),
            diagnostics={
                name: tuple(edge.qc.diagnostics[name] for edge in detections)
                for name in diagnostic_fields
            },
            extras={} if extras is None else extras,
        )

    def to_dict(self) -> dict:
        """Return JSON data with all standard fields explicitly present."""
        return {
            "check_name": self.check_name,
            "frame_indices": list(self.frame_indices),
            "rejected": list(self.rejected),
            "diagnostics": _json_value(self.diagnostics),
            "trajectory_flags": sorted(self.trajectory_flags),
            "extras": _json_value(self.extras),
        }

    @classmethod
    def from_dict(cls, values: dict) -> QCResult:
        """Load the standard result structure and validate its contents."""
        fields = {"check_name", "frame_indices", "rejected", "diagnostics",
                  "trajectory_flags", "extras"}
        if not isinstance(values, dict) or set(values) != fields:
            raise ValueError("QCResult data must contain exactly the standard fields.")
        return cls(**values)
