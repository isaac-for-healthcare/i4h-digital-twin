# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone OpenUSD snapshots of posed SOMA skin and named internal anatomy."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np


def export_to_usd(body, path, *, skin_opacity=0.15, pose="scan"):
    """Export a self-contained patient, supine with arms down by default.

    Meshes and their rigid poses are relative to the SOMA human origin; the
    root places the head along +X and anterior along +Z in a meter/Z-up stage.
    ``pose="current"`` retains the displayed pose and upright axis conversion.
    CT is optional, stored as HU in ZYX order with a voxel-to-human matrix.
    Centerlines already extracted on structures share their mesh's local frame.
    Export does not change the live body's pose or visibility.
    """
    from ..geometry import transform_points
    from ..human import HumanBody
    from ..soma_body import PosedBody
    from ..structures import AnatomicalStructure

    if pose not in {"scan", "current"}:
        raise ValueError("pose must be scan or current")
    if body.soma is None or body.soma.soma_body is None:
        ct = ct_to_human = None
        if body.imaging is not None:
            from .utils import attached_ct

            ct = attached_ct(body.imaging)
            lps_from_body = (
                np.diag([-1.0, -1.0, 1.0, 1.0]) @ body.imaging.body_to_imaging
            )
            ct_to_human = np.linalg.inv(lps_from_body)
        return _export_to_usd(
            body,
            path,
            skin_opacity=skin_opacity,
            root_transform=np.eye(4),
            ct=ct,
            ct_to_human=ct_to_human,
        )
    skin = body.soma.soma_body
    if pose == "scan" and body.soma.soma_layer is not None:
        import torch

        parameters = body.soma.soma_parameters
        reference = parameters["poses"]
        parameters["poses"] = torch.as_tensor(
            body.soma.scan_pose, dtype=reference.dtype, device=reference.device
        )
        with torch.no_grad():
            skin = PosedBody.from_output(
                body.soma.soma_layer, body.soma.soma_layer.forward(**parameters)
            )
    human_from_soma = np.linalg.inv(skin.transforms.get("Root", np.eye(4)))

    def placement(structure):
        if pose == "scan" and structure.local_to_anchor is not None:
            return skin.transforms[structure.anchor_joint] @ structure.local_to_anchor
        return structure.local_to_world

    snapshot = HumanBody(
        {
            name: AnatomicalStructure(
                name,
                structure.kind,
                structure.mesh.vertices,
                structure.mesh.faces,
                local_to_world=human_from_soma @ placement(structure),
                anchor_joint=structure.anchor_joint,
                enabled=structure.enabled,
                centerline=structure.centerline,
            )
            for name, structure in body.anatomy.structures.items()
        }
    )
    from ..soma_body import SomaRepresentation

    snapshot.soma = SomaRepresentation(snapshot.anatomy)
    snapshot.soma.soma_body = PosedBody(
        transform_points(skin.vertices, human_from_soma), skin.faces, {}, {}
    )
    ct = ct_to_human = None
    if body.imaging is not None:
        from .utils import attached_ct

        ct = attached_ct(body.imaging)
        lps_from_body = np.diag([-1.0, -1.0, 1.0, 1.0]) @ body.imaging.body_to_imaging
        ct_to_human = (
            human_from_soma @ body.soma.body_to_soma @ np.linalg.inv(lps_from_body)
        )
    root_transform = None
    if pose == "scan":
        root_transform = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return _export_to_usd(
        snapshot,
        path,
        skin_opacity=skin_opacity,
        root_transform=root_transform,
        ct=ct,
        ct_to_human=ct_to_human,
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
        f"{Path(path).name}: /HumanBody (default Xform, meters, Z-up)\n"
        f"  Exterior/SOMA: SOMA-exported skin mesh\n"
        f"  Anatomy: {len(anatomy)} meshes ({hidden} hidden), {centerlines} centerlines\n"
        f"  Looks: UsdPreviewSurface materials\n"
        + (
            f"  Imaging/CT: custom HU voxels {tuple(ct.GetAttribute('ct:shapeZYX').Get())}, "
            "voxel-to-human transform\n"
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
    skin_name="SOMA",
    ct=None,
    ct_to_human=None,
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

    def mesh(
        prim_path,
        vertices,
        faces,
        color,
        opacity,
        matrix=None,
        enabled=True,
        soma=False,
    ):
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
        if soma:
            # Delegate skin topology/points to SOMA's native static USD exporter.
            # Copy the authored prim so the final file has no sidecar references.
            from soma.io import write_usd_mesh

            with TemporaryDirectory(prefix="soma-usd-") as folder:
                source = Path(folder) / "skin.usdc"
                write_usd_mesh(
                    source,
                    "/SOMA",
                    vertices,
                    faces.astype(np.int32).ravel(),
                    np.full(len(faces), 3, dtype=np.int32),
                )
                layer = Sdf.Layer.FindOrOpen(str(source))
                if not Sdf.CopySpec(layer, "/SOMA", stage.GetRootLayer(), prim_path):
                    raise RuntimeError("Could not copy SOMA's exported mesh")
            result = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
            result.GetPrim().SetCustomDataByKey("exporter", "soma.io.write_usd_mesh")
        else:
            result = UsdGeom.Mesh.Define(stage, prim_path)
            result.CreatePointsAttr(
                Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32))
            )
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

    if body.soma is not None and body.soma.soma_body is not None:
        UsdGeom.Scope.Define(stage, "/HumanBody/Exterior")
        skin = mesh(
            "/HumanBody/Exterior/" + skin_name,
            body.soma.soma_body.vertices,
            body.soma.soma_body.faces,
            (0.72, 0.77, 0.84),
            skin_opacity,
            soma=skin_name == "SOMA",
        )
        skin.SetDisplayName(skin_name + " exterior")
    used = {skin_name}
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
        if structure.anchor_joint:
            prim.SetCustomDataByKey("anatomy:anchor_joint", structure.anchor_joint)
        if structure.centerline is not None:
            graph = structure.centerline
            prim.CreateAttribute(
                "centerline:points", Sdf.ValueTypeNames.Point3fArray, custom=True
            ).Set(Vt.Vec3fArray.FromNumpy(np.asarray(graph.points, dtype=np.float32)))
            prim.CreateAttribute(
                "centerline:edges", Sdf.ValueTypeNames.Int2Array, custom=True
            ).Set(Vt.Vec2iArray.FromNumpy(np.asarray(graph.edges, dtype=np.int32)))
            prim.CreateAttribute(
                "centerline:radii", Sdf.ValueTypeNames.FloatArray, custom=True
            ).Set(Vt.FloatArray.FromNumpy(np.asarray(graph.radii, dtype=np.float32)))
            prim.CreateAttribute(
                "centerline:coordinateFrame", Sdf.ValueTypeNames.Token, custom=True
            ).Set("structure_local_m")
    if ct is not None:
        hu = np.asarray(ct.hu_zyx, dtype=np.float32)
        if hu.ndim != 3 or not hu.size or not np.isfinite(hu).all():
            raise ValueError("Expected a non-empty, finite 3D CT volume")
        voxel_to_lps = np.eye(4)
        voxel_to_lps[:3, :3] = np.asarray(ct.direction).reshape(3, 3) @ np.diag(
            np.asarray(ct.spacing_zyx_mm)[::-1] * 0.001
        )
        voxel_to_lps[:3, 3] = np.asarray(ct.origin_xyz_mm) * 0.001
        UsdGeom.Scope.Define(stage, "/HumanBody/Imaging")
        volume = UsdGeom.Scope.Define(stage, "/HumanBody/Imaging/CT").GetPrim()
        volume.CreateAttribute("ct:hu", Sdf.ValueTypeNames.FloatArray, custom=True).Set(
            Vt.FloatArray.FromNumpy(hu.ravel(order="C"))
        )
        volume.CreateAttribute("ct:shapeZYX", Sdf.ValueTypeNames.Int3, custom=True).Set(
            Gf.Vec3i(*hu.shape)
        )
        volume.CreateAttribute(
            "ct:voxelToHuman", Sdf.ValueTypeNames.Matrix4d, custom=True
        ).Set(Gf.Matrix4d((ct_to_human @ voxel_to_lps).T.tolist()))
        volume.CreateAttribute(
            "ct:arrayOrder", Sdf.ValueTypeNames.Token, custom=True
        ).Set("ZYX_C")
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
