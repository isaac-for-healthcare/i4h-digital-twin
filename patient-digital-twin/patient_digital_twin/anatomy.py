# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anatomy model: the organs, vessels, and systems inside a patient.

Main contents:

- ``Kind`` / ``System``: string enums for a structure's category (organ, vessel, ...)
  and the body systems it belongs to.
- ``CATALOG``: canonical structure name -> ``(Kind, frozenset[System])``. Application
  metadata, not a clinical reference ontology. ``canonical_name`` maps backend label
  spellings ("Left Kidney", "rib 3 R") onto these names; ``is_vessel`` identifies
  structures that get navigation centerlines.
- ``MeshGeometry``: stored local vertices and faces, retained while a structure is hidden.
- ``AnatomicalStructure``: one named mesh in local XYZ meters, with a rigid
  ``local_to_body`` (imported placement) and ``local_to_world`` (current pose), an
  ``enabled`` flag that hides geometry without deleting it, and an optional centerline.
- ``AnatomyCollection``: ``structures`` keyed by name, the source registration
  ``body_to_imaging``, the source label volume (for scan-grid vessel masks), plus
  ``select`` and the ``set_*_enabled`` visibility controls. Importers return one of these;
  ``HumanBody`` (in ``body.py``) wraps it.

Example::

    anatomy = SegmentationImporter("masks/", names=["aorta", "liver"]).to_anatomy_collection()
    anatomy.set_system_enabled("digestive", False)  # Hide the liver, keep its mesh.
    vessels = anatomy.select(kind=Kind.VESSEL, include_empty=False)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from .geometry import CenterlineGraph, rigid_transform, transform_points


class Kind(str, Enum):
    """Category of an anatomical structure; values are written to USD and bundle manifests."""

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
    """Body system names, used by ``AnatomyCollection.select`` and ``set_system_enabled``."""

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


def _catalog() -> dict[str, tuple[Kind, frozenset[System]]]:
    """name -> (Kind, systems). '{s}' expands to left/right; '{1-12}' to a numeric range."""
    K, S = Kind, System
    rows = [
        ("liver gallbladder stomach duodenum small_bowel colon esophagus", K.ORGAN, [S.DIGESTIVE]),
        ("pancreas", K.ORGAN, [S.DIGESTIVE, S.ENDOCRINE]),
        ("spleen", K.ORGAN, [S.LYMPHATIC_IMMUNE]),
        ("kidney_{s}", K.ORGAN, [S.URINARY, S.ENDOCRINE]),
        ("adrenal_gland_{s} thyroid_gland", K.ORGAN, [S.ENDOCRINE]),
        ("urinary_bladder", K.ORGAN, [S.URINARY]),
        ("prostate", K.ORGAN, [S.REPRODUCTIVE]),
        ("trachea", K.AIRWAY, [S.RESPIRATORY]),
        ("brain spinal_cord", K.ORGAN, [S.NERVOUS]),
        ("heart", K.ORGAN, [S.CARDIOVASCULAR]),
        ("atrial_appendage_left", K.ORGAN_PART, [S.CARDIOVASCULAR]),
        ("lung_{s}", K.ORGAN, [S.RESPIRATORY]),
        ("lung_upper_lobe_left lung_lower_lobe_left lung_upper_lobe_right "
         "lung_middle_lobe_right lung_lower_lobe_right", K.ORGAN_PART, [S.RESPIRATORY]),
        ("kidney_cyst_{s}", K.FINDING, [S.URINARY]),
        ("aorta pulmonary_vein brachiocephalic_trunk superior_vena_cava inferior_vena_cava "
         "subclavian_artery_{s} common_carotid_artery_{s} brachiocephalic_vein_{s} "
         "iliac_artery_{s} iliac_vena_{s}", K.VESSEL, [S.CARDIOVASCULAR]),
        # This segmentation label explicitly merges two different veins.
        ("portal_vein_and_splenic_vein", K.GROUP, [S.CARDIOVASCULAR]),
        # The source 'hip' label denotes the hip bone, not the joint.
        ("humerus_{s} scapula_{s} clavicula_{s} femur_{s} hip_{s} sacrum sternum "
         "vertebrae_C{1-7} vertebrae_T{1-12} vertebrae_L{1-6} vertebrae_S1 "
         "rib_left_{1-12} rib_right_{1-12}", K.BONE, [S.SKELETAL]),
        ("skull vertebrae intervertebral_discs costal_cartilages", K.GROUP, [S.SKELETAL]),
        ("gluteus_maximus_{s} gluteus_medius_{s} gluteus_minimus_{s}", K.MUSCLE, [S.MUSCULAR]),
        ("autochthon_{s} iliopsoas_{s}", K.GROUP, [S.MUSCULAR]),
    ]
    result = {}
    for names, kind, systems in rows:
        for name in names.split():
            if "{s}" in name:
                expanded = [name.replace("{s}", side) for side in ("left", "right")]
            elif match := re.search(r"\{(\d+)-(\d+)\}", name):
                low, high = int(match[1]), int(match[2])
                expanded = [name[: match.start()] + str(n) for n in range(low, high + 1)]
            else:
                expanded = [name]
            result.update({n: (kind, frozenset(systems)) for n in expanded})
    return result


CATALOG = _catalog()


def is_vessel(name: str, kind: Kind | None = None) -> bool:
    """True for vessel kinds and the merged ``portal_vein_and_splenic_vein`` label.

    ``kind`` defaults to the catalog kind of ``name``. Bundles use this to choose
    which structures get scan-grid vessel masks and navigation centerlines.
    """
    kind = CATALOG.get(name, (None,))[0] if kind is None else kind
    return kind == Kind.VESSEL or name == "portal_vein_and_splenic_vein"


def canonical_name(raw: str) -> str:
    """Normalize a backend label name to catalog spelling.

    Lower-cases, joins words with underscores, moves left/right (or l/r) to the end
    (ribs become ``rib_<side>_<n>``), and upper-cases vertebra letters.

    Example:
        ``canonical_name("Left Kidney") == "kidney_left"``; ``canonical_name("rib 3 R") == "rib_right_3"``
    """
    parts = [part for part in re.split(r"[^a-z0-9]+", raw.lower()) if part]
    side = next((p for p in parts if p in ("left", "right", "l", "r")), None)
    if side:
        parts.remove(side)
        side = {"l": "left", "r": "right"}.get(side, side)
    name = "_".join(parts)
    if side and re.fullmatch(r"rib_\d+", name):
        return f"rib_{side}_{name[4:]}"
    name += f"_{side}" if side else ""
    name = {"bladder": "urinary_bladder"}.get(name, name)
    return re.sub(r"^vertebrae_([ctls])(\d+)$", lambda m: f"vertebrae_{m[1].upper()}{m[2]}", name)


@dataclass
class MeshGeometry:
    """Stored local XYZ-meter vertices (N, 3) and triangle indices (F, 3), kept while disabled."""

    vertices: np.ndarray | None = field(default=None, repr=False)
    faces: np.ndarray | None = field(default=None, repr=False)


class AnatomicalStructure:
    """One named structure: a local XYZ-meter mesh, rigid placements, and an optional centerline.

    Normally created by an importer. Attributes:

    - ``name`` / ``kind``: canonical name and ``Kind``.
    - ``mesh``: stored geometry, retained even while disabled.
    - ``local_to_body``: imported placement in the shared body frame (meters).
    - ``local_to_world``: current display pose; edit this to move the structure.
    - ``enabled``: when False, ``vertices``/``faces``/``*_vertices`` return None.
    - ``centerline``: optional ``CenterlineGraph`` in the mesh's local frame; replacing
      ``vertices`` or ``faces`` clears it.

    Example:
        ``AnatomicalStructure("aorta", Kind.VESSEL, vertices, faces, local_to_body=placement)``
    """

    def __init__(
        self,
        name: str,
        kind: Kind | str,
        vertices: np.ndarray | None = None,
        faces: np.ndarray | None = None,
        local_to_body: np.ndarray | None = None,
        local_to_world: np.ndarray | None = None,
        *,
        enabled: bool = True,
        centerline: CenterlineGraph | None = None,
    ) -> None:
        self.name, self.kind = name, Kind(kind)
        self.mesh = MeshGeometry(vertices, faces)
        self.centerline: CenterlineGraph | None = centerline
        self.enabled = enabled
        self.local_to_body: np.ndarray = np.eye(4) if local_to_body is None else local_to_body
        self.local_to_world: np.ndarray = self.local_to_body.copy() if local_to_world is None else local_to_world

    def __repr__(self) -> str:
        return f"AnatomicalStructure({self.name!r}, {self.kind.value!r}, enabled={self.enabled})"

    @property
    def is_empty(self) -> bool:
        """True when disabled or when no geometry was imported."""
        return self.vertices is None

    @property
    def vertices(self) -> np.ndarray | None:
        """Local XYZ-meter vertices, or None while disabled or empty."""
        return self.mesh.vertices if self.enabled else None

    @vertices.setter
    def vertices(self, value: np.ndarray | None) -> None:
        self.mesh.vertices, self.centerline = value, None

    @property
    def faces(self) -> np.ndarray | None:
        """Triangle vertex indices, or None while disabled or empty."""
        return self.mesh.faces if self.enabled and self.mesh.vertices is not None else None

    @faces.setter
    def faces(self, value: np.ndarray | None) -> None:
        self.mesh.faces, self.centerline = value, None

    @property
    def body_vertices(self) -> np.ndarray | None:
        """Vertices in the shared body frame (``local_to_body``), unaffected by posing."""
        vertices = self.vertices
        return None if vertices is None else transform_points(vertices, self.local_to_body)

    @property
    def world_vertices(self) -> np.ndarray | None:
        """Vertices in the current pose (``local_to_world``); do not transform them again."""
        vertices = self.vertices
        return None if vertices is None else transform_points(vertices, self.local_to_world)


class AnatomyCollection:
    """Named structures plus the source registration they were imported with.

    Attributes:
        structures: ``dict[name, AnatomicalStructure]`` (shared, not copied).
        body_to_imaging: Optional rigid body-frame-to-RAS-meter transform (validated on set).
    """

    def __init__(self, structures: dict[str, AnatomicalStructure] | None = None) -> None:
        self.structures: dict[str, AnatomicalStructure] = {} if structures is None else structures
        self._body_to_imaging: np.ndarray | None = None

    @property
    def body_to_imaging(self) -> np.ndarray | None:
        """Optional rigid body-to-RAS-meter source registration."""
        return self._body_to_imaging

    @body_to_imaging.setter
    def body_to_imaging(self, value: ArrayLike | None) -> None:
        self._body_to_imaging = None if value is None else rigid_transform(value)

    @classmethod
    def from_names(cls, names: Iterable[str], *, strict: bool = True) -> AnatomyCollection:
        """Create empty structures (kind from the catalog) for canonical names.

        Duplicates collapse to one structure. With ``strict=False`` unknown names
        become ``Kind.UNKNOWN`` instead of raising ``ValueError``.
        """
        names = tuple(dict.fromkeys(names))
        unknown = set(names) - CATALOG.keys()
        if strict and unknown:
            raise ValueError(f"Unmapped labels: {sorted(unknown)}")
        return cls({n: AnatomicalStructure(n, CATALOG.get(n, (Kind.UNKNOWN,))[0]) for n in names})

    def set_enabled(self, enabled: bool) -> None:
        """Show or hide every structure; geometry is kept. Later calls override earlier ones."""
        for name in self.structures:
            self.set_structure_enabled(name, enabled)

    def set_system_enabled(self, system: System | str, enabled: bool) -> None:
        """Show or hide the current members of a system, e.g. ``set_system_enabled("skeletal", False)``."""
        for structure in self.select(system=system):
            self.set_structure_enabled(structure.name, enabled)

    def set_structure_enabled(self, name: str, enabled: bool) -> None:
        """Show or hide one structure by name (``KeyError`` if unknown, ``ValueError`` if not a bool)."""
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        self.structures[name].enabled = enabled

    def select(
        self, *, kind: Kind | None = None, system: System | str | None = None, include_empty: bool = True
    ) -> list[AnatomicalStructure]:
        """Structures filtered by ``kind`` and catalog ``system``, in insertion order.

        Example:
            ``body.anatomy.select(kind=Kind.VESSEL, include_empty=False)`` returns visible vessels.
        """
        selected_system = None if system is None else System(system)
        return [
            s for s in self.structures.values()
            if (kind is None or s.kind == kind)
            and (selected_system is None or selected_system in CATALOG.get(s.name, (None, frozenset()))[1])
            and (include_empty or not s.is_empty)
        ]
