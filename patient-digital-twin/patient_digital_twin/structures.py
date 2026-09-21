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

from .geometry import transform_points

if TYPE_CHECKING:
    from .topology import CenterlineGraph


class Kind(str, Enum):
    """Segmentation category; use enum values when filtering with body.select()."""

    ORGAN = "organ"
    BONE = "bone"
    MUSCLE = "muscle"
    VESSEL = "vessel"
    AIRWAY = "airway"
    ORGAN_PART = "organ_part"
    GROUP = "structure_group"
    CARTILAGE = "cartilage"
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
    INTEGUMENTARY = "integumentary"


@dataclass
class MeshGeometry:
    """Stored local XYZ-meter vertices and triangle indices."""

    vertices: np.ndarray | None = field(default=None, repr=False)
    faces: np.ndarray | None = field(default=None, repr=False)


@dataclass(init=False)
class AnatomicalStructure:
    """One anatomy item with source geometry and optional articulation.

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
    anchor_joint: str | None
    local_to_anchor: np.ndarray | None = field(repr=False)

    def __init__(
        self,
        name,
        kind,
        vertices=None,
        faces=None,
        local_to_body=None,
        local_to_world=None,
        anchor_joint=None,
        local_to_anchor=None,
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
        self.anchor_joint = anchor_joint
        self.local_to_anchor = local_to_anchor

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
        """Replace source vertices; rebind SOMA afterwards if needed."""
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

    @property
    def is_organ(self) -> bool:
        """Biological grouping that includes specialized BONE and MUSCLE kinds."""
        return self.kind in {Kind.ORGAN, Kind.BONE, Kind.MUSCLE, Kind.AIRWAY}
