# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import named STL/OBJ/OpenUSD meshes with optional rigid body-frame placement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geometry import rigid_transform, transform_points
from ._labels import body_from_labels
from ._segmentation import canonical_name


def _load_mesh(path):
    """Read file-local XYZ meters; flatten authored USD transforms and units."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".usd", ".usda", ".usdc"):
        try:
            from pxr import Usd, UsdGeom
        except ImportError as exc:
            raise ImportError("OpenUSD is optional; run: pip install usd-core") from exc
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise ValueError(f"Cannot open USD stage: {path}")
        cache = UsdGeom.XformCache()
        unit = UsdGeom.GetStageMetersPerUnit(stage)
        vertices, faces, offset = [], [], 0
        # Body/imaging XYZ uses Z as its reference up axis. Account for authored
        # USD Y-up assets before applying the caller's file-to-body placement.
        up = str(UsdGeom.GetStageUpAxis(stage))
        axis = np.eye(4)
        if up == "Y":
            axis[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
            if points.ndim != 2 or points.shape[1] != 3 or counts.sum() != len(indices):
                raise ValueError(f"Invalid USD mesh topology: {prim.GetPath()}")
            if np.any(counts != 3):
                raise ValueError(
                    f"Triangulate USD mesh before import: {prim.GetPath()}"
                )
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
            points = transform_points(points, matrix) * unit
            triangles = indices.reshape(-1, 3)
            reverse = str(mesh.GetOrientationAttr().Get()) == "leftHanded"
            if reverse != (np.linalg.det(matrix[:3, :3]) < 0):
                triangles = triangles[:, ::-1]
            vertices.append(transform_points(points, axis))
            faces.append(triangles + offset)
            offset += len(points)
        if not vertices:
            raise ValueError(f"No mesh geometry in {path}")
        return np.concatenate(vertices), np.concatenate(faces)
    if path.suffix.lower() not in (".stl", ".obj"):
        raise ValueError("Supported mesh formats: .stl, .obj, .usd, .usda, .usdc")
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "STL/OBJ import is optional; run: pip install trimesh"
        ) from exc
    scene = trimesh.load(str(path), force="scene", process=False)
    mesh = scene.to_geometry()
    return np.asarray(mesh.vertices, dtype=float), np.asarray(
        mesh.faces, dtype=np.int64
    )


class SimpleImporter:
    """Load any named catalog subset from local mesh files, without scaling anatomy.

    STL/OBJ vertices must be XYZ meters. USD authored units/transforms are baked
    into its vertices. `mesh_to_body` maps each file's coordinates to the shared
    body frame. An omitted entry uses that anatomy's bundled reference placement
    (intended for centered local meshes); unknown reference placements require
    an explicit transform. No fitting, recentering, or resizing is hidden here.
    """

    def __init__(
        self, meshes, *, mesh_to_body=None, body_to_imaging=None, reference=None
    ):
        if not meshes:
            raise ValueError("Provide at least one anatomy name and mesh path")
        self.meshes = dict(meshes)
        self.transforms = dict(mesh_to_body or {})
        self.body_to_imaging = body_to_imaging
        self.reference = (
            Path(reference)
            if reference
            else Path(__file__).with_name("reference_body.json")
        )
        self.report = None

    def to_human_body(self, *, configuration=None):
        """Return a body containing only the requested structures and retained meshes."""
        names = {raw: canonical_name(raw) for raw in self.meshes}
        if len(set(names.values())) != len(names):
            raise ValueError("Duplicate canonical anatomy names")
        transforms = {
            canonical_name(name): value for name, value in self.transforms.items()
        }
        if set(transforms) - set(names.values()):
            raise ValueError("Transform supplied for an anatomy not in meshes")
        use_reference = any(name not in transforms for name in names.values())
        reference = json.loads(self.reference.read_text()) if use_reference else {}
        body = body_from_labels(names.values())
        body.body_to_imaging = (
            self.body_to_imaging
            if self.body_to_imaging is not None
            else reference.get("body_to_imaging")
        )
        # Reference landmarks apply only when every placement uses that frame.
        if not transforms:
            body.landmarks = {
                name: np.asarray(point, dtype=float)
                for name, point in reference.get("landmarks", {}).items()
            }
        for raw, name in names.items():
            placement = transforms.get(
                name, reference.get("mesh_to_body", {}).get(name)
            )
            if placement is None:
                raise ValueError(
                    f"No sample placement for {name}; supply mesh_to_body[{name!r}]"
                )
            vertices, faces = _load_mesh(self.meshes[raw])
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not len(vertices)
                or not np.isfinite(vertices).all()
                or faces.ndim != 2
                or faces.shape[1] != 3
                or not len(faces)
                or np.any(faces < 0)
                or np.any(faces >= len(vertices))
            ):
                raise ValueError(f"Invalid mesh for {name}")
            structure = body.structures[name]
            structure.vertices, structure.faces = vertices, faces
            structure.local_to_body = rigid_transform(placement)
            structure.local_to_world = structure.local_to_body.copy()
        if configuration is not None:
            body.configure_anatomy(configuration)
        self.report = {
            "backend": "simple",
            "present": sorted(body.structures),
            "absent": [],
            "unsupported": [],
        }
        return body
