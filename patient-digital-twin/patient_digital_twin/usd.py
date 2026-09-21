# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone OpenUSD snapshots of posed SOMA skin and named internal anatomy."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def export_to_usd(body, path, *, skin_opacity=0.15):
    """Export a snapshot in the current SOMA pose."""
    return _export_to_usd(body, path, skin_opacity=skin_opacity)


def _export_to_usd(
    body, path, *, skin_opacity=0.15, root_transform=None, skin_name="SOMA"
):
    """Write current posed geometry to USD, preserving hidden source meshes.

    The stage is Z-up, meters, with a default /HumanBody prim. SOMA's Y-up world
    is rotated into that frame once at the root. Internal meshes retain their
    own rigid transforms. This is a geometry snapshot, not an animated rig.
    """
    try:
        from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, Vt
    except ImportError as exc:
        raise ImportError(
            "USD export is optional; install with: pip install usd-core"
        ) from exc
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("Use a .usd, .usda, or .usdc output file")
    if body.soma_body is None:
        raise ValueError("Attach SOMA before exporting its exterior and anatomy")
    if not np.isfinite(skin_opacity) or not 0 <= skin_opacity <= 1:
        raise ValueError("skin_opacity must be between 0 and 1")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/HumanBody")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTransformOp().Set(
        Gf.Matrix4d(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1)
        if root_transform is None
        else Gf.Matrix4d(np.asarray(root_transform).T.tolist())
    )
    UsdGeom.Scope.Define(stage, "/HumanBody/Anatomy")
    UsdGeom.Scope.Define(stage, "/HumanBody/Looks")

    def mesh(prim_path, vertices, faces, color, opacity, matrix=None, enabled=True):
        vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or not np.isfinite(vertices).all()
            or faces.ndim != 2
            or faces.shape[1:] != (3,)
            or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0)
            or np.any(faces >= len(vertices))
        ):
            raise ValueError(f"Invalid triangle mesh: {prim_path}")
        result = UsdGeom.Mesh.Define(stage, prim_path)
        result.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        result.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
        )
        result.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel())
        )
        result.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        result.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        result.CreateDoubleSidedAttr(True)
        result.CreateExtentAttr(
            Vt.Vec3fArray.FromNumpy(
                np.array([vertices.min(0), vertices.max(0)], dtype=np.float32)
            )
        )
        result.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        result.CreateDisplayOpacityAttr([opacity])
        result.CreateVisibilityAttr(
            UsdGeom.Tokens.inherited if enabled else UsdGeom.Tokens.invisible
        )
        if matrix is not None:
            from .geometry import rigid_transform

            result.AddTransformOp().Set(Gf.Matrix4d(rigid_transform(matrix).T.tolist()))
        material = UsdShade.Material.Define(
            stage, "/HumanBody/Looks/" + result.GetPrim().GetName()
        )
        shader = UsdShade.Shader.Define(
            stage, material.GetPath().AppendChild("Surface")
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(result.GetPrim()).Bind(material)
        return result.GetPrim()

    UsdGeom.Scope.Define(stage, "/HumanBody/Exterior")
    skin = mesh(
        "/HumanBody/Exterior/" + skin_name,
        body.soma_body.vertices,
        body.soma_body.faces,
        (0.72, 0.77, 0.84),
        skin_opacity,
    )
    skin.SetDisplayName(skin_name + " exterior")
    used = {skin_name}
    for name, structure in body.structures.items():
        if structure.mesh.vertices is None:
            continue
        identifier = Tf.MakeValidIdentifier(name)
        candidate, number = identifier, 2
        while candidate in used:
            candidate = f"{identifier}_{number}"
            number += 1
        used.add(candidate)
        digest = hashlib.sha256(name.encode()).digest()
        color = tuple(0.25 + 0.65 * value / 255 for value in digest[:3])
        if structure.kind.value == "bone":
            color = (0.88, 0.84, 0.69)
        prim = mesh(
            "/HumanBody/Anatomy/" + candidate,
            structure.mesh.vertices,
            structure.mesh.faces,
            color,
            1.0,
            structure.local_to_world,
            structure.enabled,
        )
        prim.SetDisplayName(name)
        prim.SetCustomDataByKey("anatomy:name", name)
        prim.SetCustomDataByKey("anatomy:kind", structure.kind.value)
        if structure.anchor_joint:
            prim.SetCustomDataByKey("anatomy:anchor_joint", structure.anchor_joint)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Finish the layer before replacing a previous export.
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        if not stage.GetRootLayer().Export(str(temporary)):
            raise RuntimeError(f"USD export failed: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
