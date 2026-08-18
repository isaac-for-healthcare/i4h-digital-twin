# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenUSD export for anatomical meshes."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def _prim_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"Mesh_{cleaned}"
    return cleaned


def _color(name: str) -> Gf.Vec3f:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return Gf.Vec3f(*(0.25 + channel / 510.0 for channel in digest[:3]))


def write_usd(
    meshes: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
) -> Path:
    """Write named triangle meshes to one OpenUSD stage.

    Vertices are interpreted as millimeters; the stage records
    ``metersPerUnit = 0.001``.
    """
    if not meshes:
        raise ValueError("At least one mesh is required.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Materials")

    for name, (vertices, faces) in meshes.items():
        prim_name = _prim_name(name)
        usd_mesh = UsdGeom.Mesh.Define(stage, f"/World/{prim_name}")
        usd_mesh.GetPointsAttr().Set([Gf.Vec3f(*map(float, vertex)) for vertex in vertices])
        usd_mesh.GetFaceVertexCountsAttr().Set([3] * len(faces))
        usd_mesh.GetFaceVertexIndicesAttr().Set(np.asarray(faces, dtype=np.int32).ravel().tolist())
        usd_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        material = UsdShade.Material.Define(stage, f"/World/Materials/{prim_name}_Material")
        shader = UsdShade.Shader.Define(stage, f"/World/Materials/{prim_name}_Material/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(_color(name))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()
    return output
