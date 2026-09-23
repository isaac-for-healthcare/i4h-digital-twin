# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anatomy collection and system views; no SOMA implementation dependencies."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .configuration import AnatomyConfiguration
from .geometry import rigid_transform
from .structures import AnatomicalStructure, Kind, System


class AnatomyCollection:
    """Own named structures and reversible mesh availability."""

    def __init__(self, structures=None):
        """Wrap a shared structure dictionary; this does not copy meshes or labels."""
        self.structures = {} if structures is None else structures
        self._configuration = AnatomyConfiguration()
        self.source_path = None
        self._body_to_imaging = None
        # Optional acquisition bounds for identifying cropped surface geometry.
        self.body_to_voxel = None
        self.source_shape_xyz = None

    @property
    def body_to_imaging(self):
        """Optional body-to-source physical transform, in XYZ meters."""
        return None if self._body_to_imaging is None else self._body_to_imaging.copy()

    @body_to_imaging.setter
    def body_to_imaging(self, value):
        self._body_to_imaging = None if value is None else rigid_transform(value).copy()

    @property
    def configuration(self) -> AnatomyConfiguration:
        """Current immutable policy; setters create and validate a replacement."""
        return self._configuration

    def configure(self, source) -> None:
        """Replace the policy atomically. Unknown structure names are errors."""
        configuration = AnatomyConfiguration.load(source)
        unknown = set(configuration.structures) - self.structures.keys()
        if unknown:
            raise ValueError(f"Unknown anatomical structures: {sorted(unknown)}")
        self._configuration = configuration
        self.apply_configuration()

    def apply_configuration(self) -> None:
        """Apply current settings, including to labels imported after configuration."""
        for structure in self.structures.values():
            structure.enabled = self._configuration.allows(structure)

    def set_enabled(self, enabled: bool) -> None:
        """Toggle the master switch while retaining system and structure choices."""
        self.configure(replace(self._configuration, enabled=enabled))

    def set_system_enabled(self, system: System | str, enabled: bool) -> None:
        """Change one system rule; explicit structure rules override it."""
        settings = {**self._configuration.systems, System(system): enabled}
        self.configure(replace(self._configuration, systems=settings))

    def set_structure_enabled(self, name: str, enabled: bool) -> None:
        """Override a known structure without changing other policy settings."""
        settings = {**self._configuration.structures, name: enabled}
        self.configure(replace(self._configuration, structures=settings))

    def select(
        self,
        *,
        kind: Kind | None = None,
        system: System | str | None = None,
        region: str | None = None,
        include_empty: bool = True,
    ) -> list[AnatomicalStructure]:
        """Filter by kind and name-based catalog membership; include_empty=False returns only active meshes."""
        from .catalog import matching_hints, structure_systems

        if system is not None:
            system = System(system)
        return [
            s
            for s in self.structures.values()
            if (kind is None or s.kind == kind)
            and (system is None or system in structure_systems(s.name))
            and (region is None or region in matching_hints(s.name)[0])
            and (include_empty or not s.is_empty)
        ]

    def system(self, name: System | str) -> AnatomicalSystem:
        """Create a live view, not a second owner or copy of the system meshes."""
        return AnatomicalSystem(System(name), self)


@dataclass(frozen=True)
class AnatomicalSystem:
    """Live view of one system, including structures shared with other systems."""

    name: System
    anatomy: AnatomyCollection

    @property
    def structures(self) -> list[AnatomicalStructure]:
        """Current members; shared-system structures appear in each relevant view."""
        return self.anatomy.select(system=self.name)

    @property
    def is_empty(self) -> bool:
        """True if no member exposes geometry, including when there are no members."""
        return all(s.is_empty for s in self.structures)

    def set_enabled(self, enabled: bool) -> None:
        """Change the system policy; explicit per-structure overrides still apply."""
        self.anatomy.set_system_enabled(self.name, enabled)
