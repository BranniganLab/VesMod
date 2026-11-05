"""Combine independent check decisions into a frame selection."""

from dataclasses import dataclass
from typing import ClassVar, Mapping

from .qc_result import QCResult


@dataclass(frozen=True, init=False)
class _FailureRule:
    checks: tuple[str, ...]
    operator: ClassVar[str]

    def __init__(self, *checks: str):
        if not checks or any(not isinstance(name, str) or not name for name in checks):
            raise ValueError("A rejection rule requires one or more check names.")
        if len(set(checks)) != len(checks):
            raise ValueError("A rejection rule cannot repeat a check name.")
        object.__setattr__(self, "checks", tuple(checks))

    def to_dict(self) -> dict:
        """Record the operation and the check names it combines."""
        return {"operator": self.operator, "checks": list(self.checks)}


class AnyFailed(_FailureRule):
    """Reject a frame when any of the named checks rejects it."""

    operator = "any"


class AllFailed(_FailureRule):
    """Reject a frame only when every named check rejects it."""

    operator = "all"


def rule_from_dict(values: dict) -> AnyFailed | AllFailed:
    """Load a rejection rule with explicit operation and check names."""
    if not isinstance(values, dict) or set(values) != {"operator", "checks"}:
        raise ValueError("A rejection rule must contain operator and checks.")
    if values["operator"] not in ("any", "all") or not isinstance(values["checks"], list):
        raise ValueError("A rejection rule requires any/all and a list of check names.")
    rule_type = AnyFailed if values["operator"] == "any" else AllFailed
    return rule_type(*values["checks"])


def rejected_frame_indices(
    results: Mapping[str, QCResult],
    rule: AnyFailed | AllFailed | None,
) -> frozenset[int]:
    """Combine frame decisions; the default rejects any enabled-check failure.

    Recording-level rejection is handled separately by VesicleQCResult.
    """
    names = tuple(results) if rule is None else rule.checks
    failures = [
        {index for index, failed in zip(
            results[name].frame_indices, results[name].rejected, strict=True,
        ) if failed}
        for name in names
    ]
    if not failures:
        return frozenset()
    combine = set.intersection if isinstance(rule, AllFailed) else set.union
    return frozenset(combine(*failures))
