# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone OpenUSD snapshots of named anatomy and attached imaging."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def export_to_usd(body, path):
    """Export anatomy in its current placement using scan units and Z-up metadata.

    CT is optional, stored as native HU with its full array-to-scan affine.
    Centerlines share their structure's local frame. Export does not change
    structure transforms or visibility.
    """
    from .native import scan_for_body, scan_from_body_m

    scan = scan_for_body(body)
    units = scan.meters_per_unit if scan is not None else 1.0
    placement = scan_from_body_m(body, scan)
    placement = placement.copy()
    placement[:3, 3] /= units
    return _export_to_usd(
        body,
        path,
        native_scan=scan,
        meters_per_unit=units,
        root_transform=placement,
    )


def summarize_usd(path):
    """Return a short hierarchy/count summary without dumping voxel arrays."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    anatomy = stage.GetPrimAtPath("/HumanBody/Anatomy").GetChildren()
    hidden = sum(
        UsdGeom.Imageable(p).ComputeVisibility() == "invisible" for p in anatomy
    )
    centerlines = sum(p.HasAttribute("centerline:points") for p in anatomy)
    ct = stage.GetPrimAtPath("/HumanBody/Imaging/CT")
    return (
        f"{Path(path).name}: /HumanBody (default Xform, Z-up, metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)})\n"
        f"  Anatomy: {len(anatomy)} meshes ({hidden} hidden), {centerlines} centerlines\n"
        f"  Looks: UsdPreviewSurface materials\n"
        + (
            f"  Imaging/CT: custom HU voxels {tuple(ct.GetAttribute('ct:shape').Get())}, "
            "array-to-scan transform\n"
            if ct
            else ""
        )
    )


def _export_to_usd(
    body,
    path,
    *,
    skin_opacity=0.15,
    root_transform=None,
    exterior_mesh=None,
    skin_name="CT",
    native_scan=None,
    meters_per_unit=1.0,
):
    """Write current posed geometry to USD, preserving hidden source meshes.

    The stage is Z-up, uses the requested spatial units, and has a default /HumanBody prim. Meshes retain
    their rigid transforms. An optional exterior mesh uses the same frame.
    This is a geometry snapshot, not an animated rig.
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
    if not np.isfinite(skin_opacity) or not 0 <= skin_opacity <= 1:
        raise ValueError("skin_opacity must be between 0 and 1")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/HumanBody")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTransformOp().Set(
        Gf.Matrix4d(1.0)
        if root_transform is None
        else Gf.Matrix4d(np.asarray(root_transform).T.tolist())
    )
    UsdGeom.Scope.Define(stage, "/HumanBody/Anatomy")
    UsdGeom.Scope.Define(stage, "/HumanBody/Looks")

    def mesh(
        prim_path,
        vertices,
        faces,
        color,
        opacity,
        matrix=None,
        enabled=True,
    ):
        vertices, faces = (
            np.asarray(vertices, dtype=float) / meters_per_unit,
            np.asarray(faces),
        )
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
            from ..geometry import rigid_transform

            matrix = rigid_transform(matrix).copy()
            matrix[:3, 3] /= meters_per_unit
            result.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))
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

    if exterior_mesh is not None:
        UsdGeom.Scope.Define(stage, "/HumanBody/Exterior")
        skin = mesh(
            "/HumanBody/Exterior/" + skin_name,
            exterior_mesh.vertices,
            exterior_mesh.faces,
            (0.72, 0.77, 0.84),
            skin_opacity,
        )
        skin.SetDisplayName(skin_name + " exterior")
    used = set()
    for name, structure in body.anatomy.structures.items():
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
        if structure.centerline is not None:
            graph = structure.centerline
            prim.CreateAttribute(
                "centerline:points", Sdf.ValueTypeNames.Point3fArray, custom=True
            ).Set(
                Vt.Vec3fArray.FromNumpy(
                    np.asarray(graph.points / meters_per_unit, dtype=np.float32)
                )
            )
            prim.CreateAttribute(
                "centerline:edges", Sdf.ValueTypeNames.Int2Array, custom=True
            ).Set(Vt.Vec2iArray.FromNumpy(np.asarray(graph.edges, dtype=np.int32)))
            prim.CreateAttribute(
                "centerline:radii", Sdf.ValueTypeNames.FloatArray, custom=True
            ).Set(
                Vt.FloatArray.FromNumpy(
                    np.asarray(graph.radii / meters_per_unit, dtype=np.float32)
                )
            )
            prim.CreateAttribute(
                "centerline:coordinateFrame", Sdf.ValueTypeNames.Token, custom=True
            ).Set("structure_local")
    if native_scan is not None:
        UsdGeom.Scope.Define(stage, "/HumanBody/Imaging")
        volume = UsdGeom.Scope.Define(stage, "/HumanBody/Imaging/CT").GetPrim()
        hu = native_scan.values
        volume.CreateAttribute("ct:hu", Sdf.ValueTypeNames.FloatArray, custom=True).Set(
            Vt.FloatArray.FromNumpy(hu.ravel(order="C"))
        )
        volume.CreateAttribute("ct:shape", Sdf.ValueTypeNames.Int3, custom=True).Set(
            Gf.Vec3i(*hu.shape)
        )
        volume.CreateAttribute(
            "ct:arrayOrder", Sdf.ValueTypeNames.Token, custom=True
        ).Set(native_scan.array_axes)
        volume.CreateAttribute(
            "ct:arrayIndexToScan", Sdf.ValueTypeNames.Matrix4d, custom=True
        ).Set(
            Gf.Matrix4d(
                np.asarray(
                    native_scan.metadata["output"]["array_index_to_world"]
                ).T.tolist()
            )
        )
        volume.CreateAttribute(
            "ct:coordinateFrame", Sdf.ValueTypeNames.Token, custom=True
        ).Set(native_scan.frame)
        volume.CreateAttribute(
            "ct:spatialUnit", Sdf.ValueTypeNames.Token, custom=True
        ).Set(native_scan.metadata["output"]["world_unit"])
        volume.CreateAttribute("ct:units", Sdf.ValueTypeNames.Token, custom=True).Set(
            "HU"
        )
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
