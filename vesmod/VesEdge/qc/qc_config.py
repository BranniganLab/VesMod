"""Generic VesEdge QC configuration envelope.

Individual QC modules own their validated configuration classes.  This
envelope only associates those values with the explicit built-in registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, init=False)
class VesicleQCConfig:
    """Validated configurations for the registered VesEdge QC checks."""

    checks: Mapping[str, object]

    def __init__(
        self,
        *,
        checks: Mapping[str, object] | None = None,
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
        object.__setattr__(self, "checks", dict(normalized))

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
        return {
            name: spec.config_to_dict(self.checks[name])
            for name, spec in _specifications().items()
        }

    @classmethod
    def from_dict(cls, values: dict) -> "VesicleQCConfig":
        """Deserialize check-keyed configuration."""
        if not isinstance(values, dict):
            raise TypeError("QC configuration must be a dictionary.")
        return _config_from_dict(values)


def _specifications():
    from . import qc_registry

    return {spec.name: spec for spec in qc_registry.QC_CHECKS}


def _config_from_dict(values: dict) -> VesicleQCConfig:
    from .qc_checks import config_from_dict

    return config_from_dict(values)

