# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify final artifacts; optionally exercise actual Blender import/export."""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from patient_digital_twin import CenterlineGraph, HumanBody, SegmentationImporter
from patient_digital_twin.agent.assets import finalize, prepare
from patient_digital_twin.scan_volume import from_array
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


@pytest.fixture
def asset(tmp_path):
    labels = np.zeros((8, 8, 8), np.uint8)
    labels[2:6, 2:6, 2:6] = 1
    affine = np.diag([.002, .003, .004, 1.])
    affine[:3, 3] = [.04, -.02, .08]
    body = HumanBody(SegmentationImporter(labels, {1: 'liver'}, affine_xyz_to_imaging_m=affine).to_anatomy_collection())
    body.attach_scan(from_array(labels, affine * np.array([1000, 1000, 1000, 1])[:, None], world_unit='mm'))
    body.anatomy.structures['liver'].centerline = CenterlineGraph(np.zeros((2, 3)), np.ones(2), np.array([[0, 1]]))
    source = body.export_to_usd(tmp_path / 'patient.usdc')
    work = tmp_path / 'work'
    prepare(source, work)
    return source, work


@pytest.mark.parametrize('deform', [False, True])
def test_commit_preserves_ct_texture_and_invalidates_changed_graph(asset, tmp_path, deform):
    source, work = asset
    edited = Usd.Stage.Open(str(work / 'input.usdc'))
    mesh = UsdGeom.Mesh(edited.GetPrimAtPath('/HumanBody/Anatomy/liver'))
    if deform:
        points = mesh.GetPointsAttr().Get()
        points[0] += Gf.Vec3f(1, 0, 0)
        mesh.GetPointsAttr().Set(points)
    else:
        # Blender float32 rounding must not replace otherwise unchanged geometry.
        points = mesh.GetPointsAttr().Get()
        points[0] += Gf.Vec3f(1e-6, 0, 0)
        mesh.GetPointsAttr().Set(points)
    texture = work / 'color.png'
    texture.write_bytes(b'texture-dependency')
    material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
    shader = UsdShade.Shader.Define(edited, material.GetPath().AppendChild('Texture'))
    shader.CreateIdAttr('UsdUVTexture')
    shader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
    edited.Export(str(work / 'edited.usdc'))
    before = source.read_bytes()
    output = finalize(work, tmp_path / 'final')
    assert source.read_bytes() == before
    stage = Usd.Stage.Open(str(output))
    original = Usd.Stage.Open(str(source))
    assert stage.GetDefaultPrim().GetPath() == original.GetDefaultPrim().GetPath()
    assert UsdGeom.GetStageMetersPerUnit(stage) == .001
    ct = '/HumanBody/Imaging/CT'
    for attr in original.GetPrimAtPath(ct).GetAttributes():
        assert stage.GetPrimAtPath(ct).GetAttribute(attr.GetName()).Get() == attr.Get()
    prim = stage.GetPrimAtPath('/HumanBody/Anatomy/liver')
    assert prim.GetCustomDataByKey('anatomy:name') == 'liver'
    assert bool(prim.GetAttribute('centerline:points')) != deform
    if not deform:
        np.testing.assert_array_equal(prim.GetAttribute('points').Get(),
                                      original.GetPrimAtPath(prim.GetPath()).GetAttribute('points').Get())
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    assert material
    path = stage.GetPrimAtPath(material.GetPath().AppendChild('Texture')).GetAttribute('inputs:file').Get()
    assert Path(path.resolvedPath).read_bytes() == texture.read_bytes()


def test_missing_mesh_never_publishes(asset, tmp_path):
    _, work = asset
    edited = Usd.Stage.Open(str(work / 'input.usdc'))
    edited.RemovePrim('/HumanBody/Anatomy/liver')
    edited.Export(str(work / 'edited.usdc'))
    with pytest.raises(ValueError, match='identity'):
        finalize(work, tmp_path / 'final')
    assert not (tmp_path / 'final').exists()


def test_real_blender_geometry_and_textured_material(asset, tmp_path):
    blender = os.environ.get('I4H_TEST_BLENDER')
    if not blender:
        pytest.skip('Set I4H_TEST_BLENDER to exercise an installed Blender')
    from patient_digital_twin.agent import blender as worker
    source, work = asset
    script = tmp_path / 'edit.py'
    script.write_text(f'''exec(compile(open({worker.__file__!r}).read(), {worker.__file__!r}, 'exec'))
open_asset({str(work)!r})
import bpy
obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH')
obj.data.vertices[0].co.x += .001
obj.data.update()
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project()
bpy.ops.object.mode_set(mode='OBJECT')
material = obj.data.materials[0]
material.use_nodes = True
bsdf = next(n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
texture = material.node_tree.nodes.new('ShaderNodeTexImage')
image = bpy.data.images.new('test_color', width=4, height=4)
image.generated_color = (0.8, 0.1, 0.2, 1)
image.filepath_raw = {str(work / 'color.png')!r}
image.file_format = 'PNG'
image.save()
texture.image = image
material.node_tree.links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])
export_asset({str(work)!r})
''')
    subprocess.run([blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python', str(script)], check=True, capture_output=True, text=True)
    output = finalize(work, tmp_path / 'final')
    stage = Usd.Stage.Open(str(output))
    original = Usd.Stage.Open(str(source))
    prim = stage.GetPrimAtPath('/HumanBody/Anatomy/liver')
    old = original.GetPrimAtPath(prim.GetPath())
    delta = np.asarray(prim.GetAttribute('points').Get()) - np.asarray(old.GetAttribute('points').Get())
    assert delta[0, 0] == pytest.approx(1., abs=1e-4)
    np.testing.assert_allclose(delta[1:], 0, atol=1e-4)
    assert not prim.GetAttribute('centerline:points')
    ct = '/HumanBody/Imaging/CT'
    assert stage.GetPrimAtPath(ct).GetAttribute('ct:hu').Get() == original.GetPrimAtPath(ct).GetAttribute('ct:hu').Get()
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    shader = material.ComputeSurfaceSource()[0]
    assert shader.GetIdAttr().Get() == 'UsdPreviewSurface'
    assert shader.GetInput('diffuseColor').HasConnectedSource()
    assert UsdGeom.PrimvarsAPI(prim).GetPrimvar('st').HasValue()
    textures = [a.Get() for p in stage.Traverse() for a in p.GetAttributes() if a.GetTypeName() == Sdf.ValueTypeNames.Asset and a.Get()]
    assert textures and all(Path(t.resolvedPath).is_file() for t in textures)


def test_bundle_regenerates_navigation_on_original_ct_grid(tmp_path):
    labels = np.zeros((30, 20, 25), np.uint8)
    labels[3:27, 8:13, 9:14] = 1
    body = HumanBody(SegmentationImporter(labels, {1: 'aorta'}).to_anatomy_collection())
    body.attach_scan(from_array(labels.transpose(2, 1, 0), np.eye(4), world_unit='mm'))
    source = body.export_patient_twin(tmp_path / 'source', vessel_names=['aorta'])
    for name in ('human_body.usdc', 'view.png', 'inference_summary.json'):
        (source.parent / name).write_bytes(b'stale output from the original run')
    work = tmp_path / 'work'
    prepare(source, work)
    edited = Usd.Stage.Open(str(work / 'input.usdc'))
    mesh = UsdGeom.Mesh(edited.GetPrimAtPath('/HumanBody/Anatomy/aorta'))
    points = mesh.GetPointsAttr().Get()
    mesh.GetPointsAttr().Set([p + Gf.Vec3f(2, 0, 0) for p in points])
    edited.Export(str(work / 'edited.usdc'))
    with pytest.raises(ValueError, match='vessel_names'):
        finalize(work, tmp_path / 'incomplete')
    assert not (tmp_path / 'incomplete').exists()
    output = finalize(work, tmp_path / 'final', vessel_names=['aorta'])
    assert not any((output.parent / name).exists() for name in
                   ('human_body.usdc', 'view.png', 'inference_summary.json'))
    assert (output.parent / 'volume.npy').read_bytes() == (source.parent / 'volume.npy').read_bytes()
    assert (output.parent / 'volume.yaml').read_bytes() == (source.parent / 'volume.yaml').read_bytes()
    np.testing.assert_allclose(np.load(output.parent / 'centerline_points.npy')[:, 0], 13., atol=1e-6)
    np.testing.assert_array_equal(np.load(output.parent / 'vessel_mask.npy'),
                                  np.roll(np.load(source.parent / 'vessel_mask.npy'), 2, axis=0))
