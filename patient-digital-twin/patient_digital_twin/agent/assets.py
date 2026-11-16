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
    """Merge edits into the original contract and atomically publish a new version."""
    from pxr import Usd, UsdGeom

    from ..export import _publish

    settings = json.loads((workspace / 'edit.json').read_text())
    source, manifest = Path(settings['source']), settings['manifest']
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
        if prim and prim.GetAttribute('userProperties:i4h_hidden').Get():
            UsdGeom.Imageable(mesh).MakeInvisible()
    if len(paths) != len(set(paths)) or set(paths) != set(settings['paths']):
        raise ValueError('Edited USD must contain each original mesh identity exactly once')
    changed = [_merge_mesh(original.GetPrimAtPath(path), mesh) for path, mesh in zip(paths, meshes)]
    _merge_materials(original, edited, meshes, paths, settings['token'])

    def write(folder):
        folder.mkdir()
        target = folder / source.name
        if manifest is not None:
            for key, relative in manifest['artifacts'].items():
                if Path(relative).is_absolute() or '..' in Path(relative).parts:
                    raise ValueError(f'Artifact must be inside the bundle: {relative}')
                destination = folder / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if key != 'anatomy_usd':
                    shutil.copy2(source.parent / relative, destination)
            target = folder / manifest['artifacts']['anatomy_usd']
            for item in manifest['anatomy']['structures'].values():
                item['enabled'] = UsdGeom.Imageable(original.GetPrimAtPath(item['prim_path'])).ComputeVisibility() != 'invisible'
            if any(changed) and 'vessel_mask' in manifest['artifacts']:
                if not vessel_names:
                    raise ValueError('Geometry/visibility changed: supply vessel_names to regenerate navigation artifacts')
                _navigation(original, manifest, folder, vessel_names)
            (folder / 'patient_twin.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
        _copy_textures(original, workspace, target.parent)
        original.Export(str(target))
        reopened = Usd.Stage.Open(str(target))
        if not reopened or str(reopened.GetDefaultPrim().GetPath()) != '/HumanBody':
            raise ValueError('Final USD failed to reopen')

    result = _publish(output, '.edited-patient-', write)
    return result / ('patient_twin.yaml' if manifest is not None else source.name)


def _merge_mesh(target, edited) -> bool:
    """Keep original geometry exactly when only appearance changed."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    stage, source = target.GetStage(), edited.GetStage()
    mesh, original = UsdGeom.Mesh(edited), UsdGeom.Mesh(target)
    units = UsdGeom.GetStageMetersPerUnit(stage)
    scale = Gf.Matrix4d().SetScale(UsdGeom.GetStageMetersPerUnit(source) / units)
    cache = UsdGeom.XformCache()
    matrix = cache.GetLocalToWorldTransform(edited) * scale * cache.GetLocalToWorldTransform(target).GetInverse()
    if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
        raise ValueError('Export triangulated meshes from Blender')
    points, faces = validate_triangles(
        transform_points(mesh.GetPointsAttr().Get(), np.array(matrix).T),
        np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), name=str(target.GetPath()))
    orientation = mesh.GetOrientationAttr().Get()
    if np.linalg.det(np.array(matrix)[:3, :3]) < 0:
        orientation = 'leftHanded' if orientation == 'rightHanded' else 'rightHanded'
    old_points = np.asarray(original.GetPointsAttr().Get())
    geometry_changed = (points.shape != old_points.shape
                        or not np.allclose(points * units, old_points * units, rtol=0, atol=1e-7)
                        or not np.array_equal(faces.ravel(), original.GetFaceVertexIndicesAttr().Get())
                        or orientation != original.GetOrientationAttr().Get())
    visible = UsdGeom.Imageable(edited).ComputeVisibility()
    visibility_changed = visible != UsdGeom.Imageable(target).ComputeVisibility()
    # Replace appearance properties; leave identity, transforms and unrelated metadata intact.
    names = {p.GetName() for prim in (target, edited) for p in prim.GetAuthoredProperties()
             if p.GetName().startswith(('primvars:', 'material:binding'))}
    if geometry_changed:
        names.update(set(UsdGeom.Mesh.GetSchemaAttributeNames()) - {'xformOpOrder', 'visibility', 'purpose'})
        for attr in target.GetAttributes():
            if attr.GetName().startswith('centerline:'):
                target.RemoveProperty(attr.GetName())
    for name in names:
        target.RemoveProperty(name)
        prop = edited.GetProperty(name)
        if prop and prop.IsAuthored():
            Sdf.CopySpec(source.GetRootLayer(), prop.GetPath(), stage.GetRootLayer(), target.GetPath().AppendProperty(name))
        attr = target.GetAttribute(name)
        if attr and attr.GetTypeName() == Sdf.ValueTypeNames.Normal3fArray and attr.Get():
            normals = np.asarray(attr.Get()) @ np.linalg.inv(np.array(matrix)[:3, :3]).T
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
            attr.Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
    if geometry_changed:
        original.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        original.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(original.GetPointsAttr().Get()))
        original.CreateOrientationAttr(orientation)
    original.CreateVisibilityAttr(visible)
    for child in target.GetChildren():
        if child.IsA(UsdGeom.Subset):
            stage.RemovePrim(child.GetPath())
    for child in edited.GetChildren():
        if child.IsA(UsdGeom.Subset):
            Sdf.CopySpec(source.GetRootLayer(), child.GetPath(), stage.GetRootLayer(), target.GetPath().AppendChild(child.GetName()))
    return geometry_changed or visibility_changed


def _merge_materials(original, edited, meshes, paths, token):
    """Copy shader networks and remap their connections and mesh/subset bindings."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    materials = [p for p in edited.Traverse() if p.IsA(UsdShade.Material)]
    root = '/HumanBody/Looks/Edited_' + token
    UsdGeom.Scope.Define(original, root)
    remap = {p.GetPath(): Sdf.Path(root + '/Material_' + str(i)) for i, p in enumerate(materials)}
    for material in materials:
        Sdf.CopySpec(edited.GetRootLayer(), material.GetPath(), original.GetRootLayer(), remap[material.GetPath()])
    remap.update({p.GetPath(): Sdf.Path(path) for p, path in zip(meshes, paths)})
    for root_path in remap.values():
        for prim in Usd.PrimRange(original.GetPrimAtPath(root_path)):
            for prop in prim.GetProperties():
                targets = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
                for before, after in remap.items():
                    targets = [p.ReplacePrefix(before, after) if p.HasPrefix(before) else p for p in targets]
                if isinstance(prop, Usd.Relationship):
                    prop.SetTargets(targets)
                elif targets:
                    prop.SetConnections(targets)


def _copy_textures(stage, workspace, destination):
    """Package texture dependencies with relative paths beside the final USD."""
    from pxr import Sdf

    for prim in stage.Traverse():
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
            texture = destination / 'textures' / name
            texture.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, texture)
            attr.Set(Sdf.AssetPath('textures/' + name))


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
