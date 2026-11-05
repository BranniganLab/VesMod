"""Generic VesEdge QC configuration envelope.

Individual checks own their settings. This module validates the registered
settings and the separate rule used to combine check decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .acceptance_rules import AnyFailed, AllFailed, rule_from_dict


@dataclass(frozen=True, init=False)
class VesicleQCConfig:
    """Check settings and the rule used to select accepted frames.

    reject_when=None rejects any enabled check failure. An explicit AnyFailed
    or AllFailed rule combines only its named enabled checks. Recording-level
    rejection always vetoes acceptance.
    """

    checks: Mapping[str, object]
    reject_when: AnyFailed | AllFailed | None

    def __init__(
        self,
        *,
        checks: Mapping[str, object] | None = None,
        reject_when: AnyFailed | AllFailed | None = None,
        **values: object,
    ) -> None:
        """Create a configuration from check-keyed or named check values."""
        if checks is not None and values:
            raise TypeError("checks cannot be combined with named QC configurations.")
        supplied = dict(checks) if checks is not None else values
        specs = _specifications()
        unknown = set(supplied) - set(specs)
        if unknown:
            raise TypeError(f"Unexpected QC configuration field: {sorted(unknown)[0]}")
        normalized = {
            name: supplied[name] if name in supplied else spec.default_config()
            for name, spec in specs.items()
        }
        for name, value in normalized.items():
            if not isinstance(value, specs[name].config_type):
                raise TypeError(
                    f"{name} must be a {specs[name].config_type.__name__}."
                )
        if reject_when is not None:
            if type(reject_when) not in (AnyFailed, AllFailed):
                raise TypeError("reject_when must be AnyFailed or AllFailed.")
            for name in reject_when.checks:
                if name not in specs:
                    raise ValueError(f"Rejection rule refers to an unknown check: {name}")
                if not specs[name].enabled(normalized[name]):
                    raise ValueError(f"Rejection rule refers to a disabled check: {name}")
        object.__setattr__(self, "checks", dict(normalized))
        object.__setattr__(self, "reject_when", reject_when)

    def get_config(self, name: str) -> object:
        """Return the typed configuration registered under ``name``."""
        try:
            return self.checks[name]
        except KeyError as error:
            raise KeyError(f"Unknown VesEdge QC check: {name}") from error

    def __getattr__(self, name: str) -> object:
        """Provide read-only named access for registered checks."""
        if name in self.checks:
            return self.checks[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, dict]:
        """Serialize configurations through their owning check specifications."""
        values = {
            name: spec.config_to_dict(self.checks[name])
            for name, spec in _specifications().items()
        }
        if self.reject_when is not None:
            values["reject_when"] = self.reject_when.to_dict()
        return values

    @classmethod
    def from_dict(cls, values: dict) -> "VesicleQCConfig":
        """Deserialize check-keyed configuration."""
        if not isinstance(values, dict):
            raise TypeError("QC configuration must be a dictionary.")
        values = dict(values)
        rule_data = values.pop("reject_when", None)
        rule = None if rule_data is None else rule_from_dict(rule_data)
        specs = _specifications()
        unknown = set(values) - set(specs)
        if unknown:
            raise TypeError(f"Unexpected QC configuration field: {sorted(unknown)[0]}")
        parsed = {
            name: (
                spec.config_type(**values[name])
                if name in values
                else spec.default_config()
            )
            for name, spec in specs.items()
        }
        return cls(checks=parsed, reject_when=rule)


def _specifications():
    from . import qc_registry

    return {spec.name: spec for spec in qc_registry.QC_CHECKS}
