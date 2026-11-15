# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edit exported patient meshes/materials without round-tripping CT through Blender."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml

from ..geometry import transform_points, validate_triangles


def prepare(source: Path, workspace: Path) -> dict:
    """Snapshot a standalone USD or schema-2 bundle; hand only scene content to Blender."""
    from pxr import Sdf, Usd, UsdGeom

    source = source.expanduser().resolve()
    if source.is_dir():
        source /= 'patient_twin.yaml'
    manifest = yaml.safe_load(source.read_text()) if source.suffix in {'.yaml', '.yml'} else None
    if manifest is not None and manifest.get('schema_version') != 2:
        raise ValueError('Expected a schema-2 patient bundle')
    usd = source if manifest is None else source.parent / manifest['artifacts']['anatomy_usd']
    stage = Usd.Stage.Open(str(usd))
    if not stage or str(stage.GetDefaultPrim().GetPath()) != '/HumanBody':
        raise ValueError('Expected a pipeline patient USD with default prim /HumanBody')
    if UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Expected pipeline Z-up geometry')
    workspace.mkdir(parents=True, exist_ok=True)
    stage.Flatten().Export(str(workspace / 'original.usdc'))
    paths = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if not paths:
        raise ValueError('The asset has no meshes')
    settings = dict(source=str(source), usd=str(usd), manifest=manifest, paths=paths, token=uuid4().hex)
    (workspace / 'edit.json').write_text(json.dumps(settings))
    handoff = Usd.Stage.Open(str(workspace / 'original.usdc'))
    handoff.RemovePrim('/HumanBody/Imaging')
    for path in paths:
        prim = handoff.GetPrimAtPath(path)
        prim.CreateAttribute('userProperties:i4h_path', Sdf.ValueTypeNames.String).Set(path)
    handoff.Export(str(workspace / 'input.usdc'))
    return settings


def finalize(workspace: Path, output: Path, *, vessel_names: list[str] | None = None) -> Path:
    """Merge supported edits into the original contract and atomically publish a new version."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    from ..export import _publish

    settings = json.loads((workspace / 'edit.json').read_text())
    source = Path(settings['source'])
    manifest = settings['manifest']
    if manifest is not None and output.resolve().is_relative_to(source.parent):
        raise ValueError('Write the edited bundle outside the source bundle')
    original = Usd.Stage.Open(str(workspace / 'original.usdc'))
    edited = Usd.Stage.Open(str(workspace / 'edited.usdc'))
    if not edited or UsdGeom.GetStageUpAxis(edited) != 'Z':
        raise ValueError('Expected a Z-up Blender USD export')
    meshes = [p for p in edited.Traverse() if p.IsA(UsdGeom.Mesh)]
    paths = []
    for mesh in meshes:
        prim = mesh
        while prim and not prim.GetAttribute('userProperties:i4h_path'):
            prim = prim.GetParent()
        paths.append(str(prim.GetAttribute('userProperties:i4h_path').Get()) if prim else '')
    if len(paths) != len(set(paths)) or set(paths) != set(settings['paths']):
        raise ValueError('Edited USD must contain each original mesh identity exactly once')
    old_cache, new_cache = UsdGeom.XformCache(), UsdGeom.XformCache()
    units = UsdGeom.GetStageMetersPerUnit(original)
    scale = Gf.Matrix4d().SetScale(UsdGeom.GetStageMetersPerUnit(edited) / units)
    # Copy materials together with their shader networks, then remap bindings/connections.
    materials = [p for p in edited.Traverse() if p.IsA(UsdShade.Material)]
    material_root = '/HumanBody/Looks/Edited_' + settings['token']
    UsdGeom.Scope.Define(original, material_root)
    remap = {p.GetPath(): Sdf.Path(material_root + '/Material_' + str(i)) for i, p in enumerate(materials)}
    remap.update({p.GetPath(): Sdf.Path(path) for p, path in zip(meshes, paths)})
    changed = []
    for prim, path in zip(meshes, paths):
        old = original.GetPrimAtPath(path)
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError('Export triangulated meshes from Blender')
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        matrix = new_cache.GetLocalToWorldTransform(prim) * scale * old_cache.GetLocalToWorldTransform(old).GetInverse()
        points = transform_points(points, np.array(matrix).T)
        points, faces = validate_triangles(points, faces, name=path)
        old_mesh = UsdGeom.Mesh(old)
        old_points = np.asarray(old_mesh.GetPointsAttr().Get())
        geometry_changed = (points.shape != old_points.shape
                            or not np.allclose(points * units, old_points * units, rtol=0, atol=1e-7)
                            or not np.array_equal(faces.ravel(), old_mesh.GetFaceVertexIndicesAttr().Get()))
        metadata, display_name = old.GetCustomData(), old.GetDisplayName()
        graph = [(a.GetName(), a.GetTypeName(), a.Get()) for a in old.GetAttributes()
                 if a.GetName().startswith('centerline:')]
        pose = UsdGeom.Xformable(old).GetLocalTransformation()
        visible = UsdGeom.Imageable(prim).ComputeVisibility()
        visibility_changed = visible != UsdGeom.Imageable(old).ComputeVisibility()
        original.RemovePrim(path)
        Sdf.CopySpec(edited.GetRootLayer(), prim.GetPath(), original.GetRootLayer(), path)
        target = original.GetPrimAtPath(path)
        target.SetCustomData(metadata)
        target.SetDisplayName(display_name)
        target.RemoveProperty('userProperties:i4h_path')
        for attr in target.GetAttributes():
            if attr.GetName().startswith(('xformOp:', 'centerline:')):
                target.RemoveProperty(attr.GetName())
        xform = UsdGeom.Xformable(target)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(pose)
        result = UsdGeom.Mesh(target)
        result.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        result.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(result.GetPointsAttr().Get()))
        result.CreateVisibilityAttr(visible)
        if np.linalg.det(np.array(matrix)[:3, :3]) < 0:
            orientation = result.GetOrientationAttr().Get()
            result.CreateOrientationAttr('leftHanded' if orientation == 'rightHanded' else 'rightHanded')
        # Normals are directions in the mesh's local frame, unlike points.
        for attr in target.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Normal3fArray and attr.Get():
                normals = np.asarray(attr.Get()) @ np.linalg.inv(np.array(matrix)[:3, :3]).T
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
                attr.Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        if geometry_changed or visibility_changed:
            changed.append(path)
        if not geometry_changed:
            for name, kind, value in graph:
                target.CreateAttribute(name, kind, custom=True).Set(value)
    for material in materials:
        Sdf.CopySpec(edited.GetRootLayer(), material.GetPath(), original.GetRootLayer(), remap[material.GetPath()])
    for root in remap.values():
        for prim in Usd.PrimRange(original.GetPrimAtPath(root)):
            for prop in prim.GetProperties():
                targets = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
                for before, after in remap.items():
                    targets = [p.ReplacePrefix(before, after) if p.HasPrefix(before) else p for p in targets]
                if isinstance(prop, Usd.Relationship):
                    prop.SetTargets(targets)
                elif targets:
                    prop.SetConnections(targets)

    def write(folder):
        if manifest is None:
            folder.mkdir()
            target = folder / source.name
        else:
            shutil.copytree(source.parent, folder)
            target = folder / manifest['artifacts']['anatomy_usd']
            for item in manifest['anatomy']['structures'].values():
                item['enabled'] = UsdGeom.Imageable(original.GetPrimAtPath(item['prim_path'])).ComputeVisibility() != 'invisible'
            if changed and 'vessel_mask' in manifest['artifacts']:
                if not vessel_names:
                    raise ValueError('Geometry changed: supply vessel_names to regenerate the navigation artifacts')
                _navigation(original, manifest, folder, vessel_names)
            (folder / 'patient_twin.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
        # Localize texture dependencies from both source and edited shader networks.
        for prim in original.Traverse():
            for attr in prim.GetAttributes():
                if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                asset = attr.Get()
                if not asset or not asset.path:
                    continue
                path = Path(asset.resolvedPath or asset.path)
                if not path.is_absolute():
                    path = workspace / path
                if not path.is_file():
                    raise ValueError(f'Unresolved texture: {asset.path}')
                name = hashlib.sha256(path.read_bytes()).hexdigest()[:16] + path.suffix
                texture = target.parent / 'textures' / name
                texture.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, texture)
                attr.Set(Sdf.AssetPath('textures/' + name))
        original.Export(str(target))
        reopened = Usd.Stage.Open(str(target))
        if not reopened or str(reopened.GetDefaultPrim().GetPath()) != '/HumanBody':
            raise ValueError('Final USD failed to reopen')

    result = _publish(output, '.edited-patient-', write)
    return result / ('patient_twin.yaml' if manifest is not None else source.name)


def _navigation(stage, manifest, folder, names):
    """Reuse existing voxelization/centerline code; never retain stale navigation arrays."""
    from pxr import UsdGeom
    from ..geometry import native_centerline, voxelize_mesh
    from ..scan_volume import load_artifact

    artifacts = manifest['artifacts']
    scan = load_artifact(folder / artifacts['volume_metadata'])
    mask = np.zeros(scan.values_kji.shape, bool)
    cache = UsdGeom.XformCache()
    for name in names:
        prim = stage.GetPrimAtPath(manifest['anatomy']['structures'][name]['prim_path'])
        if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            raise ValueError(f'Navigation vessel is hidden: {name}')
        mesh = UsdGeom.Mesh(prim)
        world = transform_points(mesh.GetPointsAttr().Get(), np.array(cache.GetLocalToWorldTransform(prim)).T)
        indices = transform_points(world, np.linalg.inv(scan.ijk_to_world))
        mask |= voxelize_mesh(indices, np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
                              shape_zyx=mask.shape, spacing_zyx_m=(1., 1., 1.), origin_xyz_m=(0., 0., 0.))
    native = mask.transpose(['kji'.index(c) for c in scan.array_axes])
    arrays = dict(zip(('centerline_points', 'centerline_edges', 'centerline_radii'), native_centerline(native, scan)))
    arrays['vessel_mask'] = native.astype(np.uint8)
    for key, array in arrays.items():
        np.save(folder / artifacts[key], array, allow_pickle=False)
