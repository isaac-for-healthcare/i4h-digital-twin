# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated YAML mesh-availability policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

from .structures import AnatomicalStructure, System


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject ambiguous duplicate YAML keys instead of silently taking the last."""

    def construct_mapping(self, node, deep=False):
        """SafeLoader hook enforcing unique string keys; use AnatomyConfiguration.load()."""
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise TypeError("Configuration keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True)
class AnatomyConfiguration:
    """Master switch > explicit structure setting > all system settings.

    A multi-system structure is disabled if any of its systems is disabled,
    unless an explicit structure setting overrides that decision.
    Unspecified settings default to enabled. Disabling never deletes storage.
    """

    enabled: bool = True
    systems: Mapping[System, bool] = field(default_factory=dict)
    structures: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        """Validate booleans and enum names, then copy settings into read-only mappings."""
        if not isinstance(self.enabled, bool):
            raise ValueError("anatomy.enabled must be a boolean")
        for name, values in (
            ("systems", self.systems),
            ("structures", self.structures),
        ):
            if not isinstance(values, Mapping):
                raise TypeError(f"anatomy.{name} must be a mapping")
            if any(not isinstance(key, str) or not key for key in values):
                raise ValueError(f"anatomy.{name} keys must be non-empty names")
            if any(not isinstance(value, bool) for value in values.values()):
                raise ValueError(f"anatomy.{name} settings must be booleans")
        object.__setattr__(
            self,
            "systems",
            MappingProxyType(
                {System(key): value for key, value in self.systems.items()}
            ),
        )
        object.__setattr__(self, "structures", MappingProxyType(dict(self.structures)))

    @classmethod
    def load(cls, source) -> AnatomyConfiguration:
        """Read a YAML path, a mapping with an 'anatomy' section, or a policy."""
        if isinstance(source, cls):
            return source
        if isinstance(source, (str, Path)):
            with Path(source).open(encoding="utf-8") as stream:
                source = yaml.load(stream, Loader=_UniqueKeyLoader)
            if source is None:
                source = {}
        if not isinstance(source, Mapping):
            raise TypeError("Configuration must be a mapping with an anatomy section")
        extra = set(source) - {"anatomy"}
        if extra:
            raise ValueError(
                f"Unknown configuration sections: {sorted(extra, key=str)}"
            )
        anatomy = source.get("anatomy", {})
        if not isinstance(anatomy, Mapping):
            raise TypeError("anatomy must be a mapping")
        extra = set(anatomy) - {"enabled", "systems", "structures"}
        if extra:
            raise ValueError(f"Unknown anatomy settings: {sorted(extra, key=str)}")
        return cls(**anatomy)

    def allows(self, structure: AnatomicalStructure) -> bool:
        """Evaluate policy only; True does not imply a source segmentation mesh exists."""
        from .catalog import structure_systems

        if not self.enabled:
            return False
        if structure.name in self.structures:
            return self.structures[structure.name]
        return all(
            self.systems.get(system, True)
            for system in structure_systems(structure.name)
        )

    def to_dict(self) -> dict:
        """Plain mapping suitable for yaml.safe_dump."""
        return {
            "anatomy": {
                "enabled": self.enabled,
                "systems": {key.value: value for key, value in self.systems.items()},
                "structures": dict(self.structures),
            }
        }
