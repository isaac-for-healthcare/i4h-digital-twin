# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rigid anatomical structures and reversible mesh availability.

MeshGeometry retains source geometry even while a structure is empty.
The public vertices/faces properties expose only enabled geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from .geometry import rigid_transform, transform_points

if TYPE_CHECKING:
    from .topology import CenterlineGraph


class Kind(str, Enum):
    """Segmentation category; use enum values when filtering with body.anatomy.select()."""

    ORGAN = "organ"
    BONE = "bone"
    MUSCLE = "muscle"
    VESSEL = "vessel"
    AIRWAY = "airway"
    ORGAN_PART = "organ_part"
    GROUP = "structure_group"
    FINDING = "finding"
    UNKNOWN = "unknown"


class System(str, Enum):
    """Supported system names, also used verbatim as YAML systems keys."""

    SKELETAL = "skeletal"
    MUSCULAR = "muscular"
    NERVOUS = "nervous"
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    DIGESTIVE = "digestive"
    URINARY = "urinary"
    ENDOCRINE = "endocrine"
    LYMPHATIC_IMMUNE = "lymphatic_immune"
    REPRODUCTIVE = "reproductive"


@dataclass
class MeshGeometry:
    """Stored local XYZ-meter vertices and triangle indices."""

    vertices: np.ndarray | None = field(default=None, repr=False)
    faces: np.ndarray | None = field(default=None, repr=False)


@dataclass(init=False)
class AnatomicalStructure:
    """One anatomy item with source geometry and rigid placement.

    Normally created by an importer. For manual construction supply local XYZ
    vertices in meters, triangle indices, and local_to_body. Use the public
    geometry properties for display and mesh for retained disabled storage.
    centerline stores topology in the same local frame as the source mesh.
    Replacing vertices/faces clears it; after direct in-place mesh edits, call
    HumanBody.extract_topology() again to refresh it.
    """

    name: str
    kind: Kind
    mesh: MeshGeometry = field(repr=False)
    centerline: CenterlineGraph | None = field(default=None, repr=False)
    enabled: bool
    local_to_body: np.ndarray = field(repr=False)
    local_to_world: np.ndarray = field(repr=False)

    def __init__(
        self,
        name,
        kind,
        vertices=None,
        faces=None,
        local_to_body=None,
        local_to_world=None,
        *,
        enabled=True,
        centerline=None,
    ):
        """Create metadata alone or retain caller-supplied local mesh arrays.

        Arrays are not copied or automatically meshed. local_to_body describes
        the imported placement relative to the shared body origin.
        """
        self.name = name
        self.kind = Kind(kind)
        self.mesh = MeshGeometry(vertices, faces)
        self.centerline = centerline
        self.enabled = enabled
        self.local_to_body = np.eye(4) if local_to_body is None else local_to_body
        self.local_to_world = (
            self.local_to_body.copy() if local_to_world is None else local_to_world
        )

    @property
    def is_empty(self) -> bool:
        """True when disabled or when no segmentation geometry was supplied."""
        return self.vertices is None

    @property
    def vertices(self) -> np.ndarray | None:
        """Enabled source vertices in local XYZ meters, or None when empty."""
        return self.mesh.vertices if self.enabled else None

    @vertices.setter
    def vertices(self, value):
        """Replace source vertices and clear the cached centerline."""
        self.mesh.vertices = value
        self.centerline = None

    @property
    def faces(self) -> np.ndarray | None:
        """Enabled triangle indices into vertices; None when geometry is unavailable."""
        return (
            self.mesh.faces if self.enabled and self.mesh.vertices is not None else None
        )

    @faces.setter
    def faces(self, value):
        """Store triangle connectivity, including while disabled; no remeshing occurs."""
        self.mesh.faces = value
        self.centerline = None

    @property
    def body_vertices(self) -> np.ndarray | None:
        """Recover imported body-frame coordinates, unaffected by posing."""
        return (
            None
            if self.is_empty
            else transform_points(self.vertices, self.local_to_body)
        )

    @property
    def world_vertices(self) -> np.ndarray | None:
        """Render these current world coordinates; do not apply local_to_world again."""
        if self.is_empty:
            return None
        return transform_points(self.vertices, self.local_to_world)


class AnatomyCollection:
    """Own named structures and reversible mesh availability."""

    def __init__(self, structures=None):
        """Wrap a shared structure dictionary; this does not copy meshes or labels."""
        self.structures = {} if structures is None else structures
        self._body_to_imaging = None
        # Optional source label volume, used to rasterize vessels on the scan grid.
        self.source_segmentation = None
        self.source_label_names = {}
        self.source_voxel_to_ras_m = None

    @property
    def body_to_imaging(self):
        """Optional body-to-source physical transform, in XYZ meters."""
        return None if self._body_to_imaging is None else self._body_to_imaging.copy()

    @body_to_imaging.setter
    def body_to_imaging(self, value):
        self._body_to_imaging = None if value is None else rigid_transform(value).copy()

    def set_enabled(self, enabled: bool) -> None:
        """Set visibility of all structures without discarding geometry."""
        for structure in self.structures.values():
            self.set_structure_enabled(structure.name, enabled)

    def set_system_enabled(self, system: System | str, enabled: bool) -> None:
        """Set visibility of the current members of a system."""
        for structure in self.select(system=system):
            self.set_structure_enabled(structure.name, enabled)

    def set_structure_enabled(self, name: str, enabled: bool) -> None:
        """Set visibility directly; later calls take precedence."""
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        self.structures[name].enabled = enabled

    def select(
        self,
        *,
        kind: Kind | None = None,
        system: System | str | None = None,
        region: str | None = None,
        include_empty: bool = True,
    ) -> list[AnatomicalStructure]:
        """Filter by kind and name-based catalog membership; include_empty=False returns only active meshes."""
        from .catalog import structure_regions, structure_systems

        if system is not None:
            system = System(system)
        return [
            s
            for s in self.structures.values()
            if (kind is None or s.kind == kind)
            and (system is None or system in structure_systems(s.name))
            and (region is None or region in structure_regions(s.name))
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
        """Set visibility of this system’s current structures."""
        self.anatomy.set_system_enabled(self.name, enabled)
