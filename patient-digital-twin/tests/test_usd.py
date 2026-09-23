# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual USD round trips, world transforms, material bindings and hidden meshes."""

import numpy as np
import pytest
from patient_digital_twin import AnatomicalStructure, HumanBody, Kind, PosedBody

pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdShade


def body():
    vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    faces = np.array([[0, 1, 2]])
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    result = HumanBody(
        {
            "left organ": AnatomicalStructure(
                "left organ", Kind.ORGAN, vertices, faces, local_to_world=transform
            ),
            "left-organ": AnatomicalStructure(
                "left-organ", Kind.ORGAN, vertices, faces, enabled=False
            ),
            "missing": AnatomicalStructure("missing", Kind.ORGAN),
        }
    )
    from patient_digital_twin import SomaRepresentation

    result.soma = SomaRepresentation(result.anatomy)
    result.soma.soma_body = PosedBody(vertices, faces, {}, {})
    return result


@pytest.mark.parametrize("suffix", [".usd", ".usda", ".usdc"])
def test_export_roundtrip_and_independent_mesh_visibility(tmp_path, suffix):
    patient = body()
    target = patient.export_to_usd(tmp_path / ("patient" + suffix))
    stage = Usd.Stage.Open(str(target))
    assert stage.GetDefaultPrim().GetPath() == "/HumanBody"
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    organs = stage.GetPrimAtPath("/HumanBody/Anatomy").GetChildren()
    assert len(organs) == 2
    assert len({p.GetName() for p in organs}) == 2
    cache = UsdGeom.XformCache()
    for prim in organs:
        name = prim.GetCustomDataByKey("anatomy:name")
        structure = patient.anatomy.structures[name]
        mesh = UsdGeom.Mesh(prim)
        assert mesh.ComputeVisibility() == (
            "inherited" if structure.enabled else "invisible"
        )
        points = mesh.GetPointsAttr().Get()
        actual = np.array(
            [
                cache.GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(p))
                for p in points
            ]
        )
        source = (
            structure.mesh.vertices @ structure.local_to_world[:3, :3].T
            + structure.local_to_world[:3, 3]
        )
        np.testing.assert_allclose(actual, source[:, [1, 0, 2]] * [1, -1, 1], atol=1e-7)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert material
        mesh.MakeInvisible()
    skin = UsdGeom.Mesh(stage.GetPrimAtPath("/HumanBody/Exterior/SOMA"))
    assert skin.ComputeVisibility() == "inherited"
    assert skin.GetDisplayOpacityAttr().Get()[0] == pytest.approx(0.15)
    assert not stage.GetRootLayer().GetExternalReferences()
    assert patient.anatomy.structures["left organ"].enabled


def test_bad_mesh_does_not_overwrite_existing_file(tmp_path):
    target = tmp_path / "existing.usda"
    target.write_text("original")
    patient = body()
    patient.anatomy.structures["left organ"].faces = np.array([[0, 1, 999]])
    with pytest.raises(ValueError, match="Invalid triangle mesh"):
        patient.export_to_usd(target)
    assert target.read_text() == "original"
    patient.soma = None
    with pytest.raises(ValueError, match="Invalid triangle mesh"):
        patient.export_to_usd(target)
    assert target.read_text() == "original"


@pytest.mark.parametrize("pose", ["scan", "current"])
def test_root_moves_skin_and_anatomy_together(tmp_path, pose):
    patient = body()
    target = patient.export_to_usd(tmp_path / "patient.usdc", pose=pose)
    stage = Usd.Stage.Open(str(target))
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetDefaultPrim())
    head = matrix.TransformDir(Gf.Vec3d(0, 1, 0))
    anterior = matrix.TransformDir(Gf.Vec3d(0, 0, 1))
    np.testing.assert_allclose(head, [1, 0, 0] if pose == "scan" else [0, 0, 1])
    np.testing.assert_allclose(anterior, [0, 0, 1] if pose == "scan" else [0, -1, 0])
    skin = stage.GetPrimAtPath("/HumanBody/Exterior/SOMA")
    assert skin.GetCustomDataByKey("exporter") == "soma.io.write_usd_mesh"


def test_embedded_ct_and_hidden_centerline_roundtrip(tmp_path):
    import nibabel as nib
    from patient_digital_twin.topology import CenterlineGraph

    patient = body()
    graph = CenterlineGraph(
        np.array([[0, 0, 0], [0.1, 0, 0]]),
        np.array([0.01, 0.02]),
        np.array([[0, 1]]),
    )
    patient.anatomy.structures["left-organ"].centerline = graph
    hu = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    affine = np.diag([-2.0, -3.0, 4.0, 1.0])
    affine[:3, 3] = [10, 20, 30]
    ct_path = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(hu, affine), ct_path)
    affine_m = affine.copy()
    affine_m[:3] *= 0.001
    patient.AttachImaging(
        hu.transpose(2, 1, 0),
        voxel_to_imaging=affine_m,
        body_to_imaging=np.eye(4),
        source_path=ct_path,
    )
    ct_path.unlink()  # Export must consume the array, not reopen provenance.
    target = patient.export_to_usd(tmp_path / "patient.usdc")
    stage = Usd.Stage.Open(str(target))
    ct = stage.GetPrimAtPath("/HumanBody/Imaging/CT")
    assert ct.GetAttribute("ct:hu").IsCustom()
    shape = tuple(ct.GetAttribute("ct:shapeZYX").Get())
    np.testing.assert_array_equal(
        np.array(ct.GetAttribute("ct:hu").Get()).reshape(shape), hu.transpose(2, 1, 0)
    )
    matrix = ct.GetAttribute("ct:voxelToHuman").Get()
    np.testing.assert_allclose(
        matrix.Transform(Gf.Vec3d(1, 2, 3)), [0.008, 0.014, 0.042]
    )
    hidden = next(
        p
        for p in stage.GetPrimAtPath("/HumanBody/Anatomy").GetChildren()
        if p.GetCustomDataByKey("anatomy:name") == "left-organ"
    )
    np.testing.assert_allclose(
        hidden.GetAttribute("centerline:points").Get(), graph.points
    )
    np.testing.assert_array_equal(
        hidden.GetAttribute("centerline:edges").Get(), graph.edges
    )
    np.testing.assert_allclose(
        hidden.GetAttribute("centerline:radii").Get(), graph.radii
    )
    assert UsdGeom.Imageable(hidden).ComputeVisibility() == "invisible"
    assert not stage.GetRootLayer().GetExternalReferences()


def test_scan_export_does_not_mutate_displayed_pose(tmp_path):
    from test_soma import ArticulatedLayer

    patient = body()
    patient.AttachExternalBody(ArticulatedLayer(), body_to_soma=np.eye(4))
    patient.soma.pose(transl=[[5, 6, 7]])
    original_skin = patient.soma.soma_body.vertices.copy()
    original_matrices = {
        n: s.local_to_world.copy() for n, s in patient.anatomy.structures.items()
    }
    target = patient.export_to_usd(tmp_path / "scan.usdc")
    stage = Usd.Stage.Open(str(target))
    exported_skin = UsdGeom.Mesh(stage.GetPrimAtPath("/HumanBody/Exterior/SOMA"))
    np.testing.assert_allclose(
        exported_skin.GetPointsAttr().Get(),
        patient.soma.soma_layer.skin.numpy(),
        atol=1e-6,
    )
    for prim in stage.GetPrimAtPath("/HumanBody/Anatomy").GetChildren():
        structure = patient.anatomy.structures[prim.GetCustomDataByKey("anatomy:name")]
        local = np.array(UsdGeom.Xformable(prim).GetLocalTransformation()).T
        np.testing.assert_allclose(local, structure.local_to_anchor, atol=1e-6)
    np.testing.assert_array_equal(patient.soma.soma_body.vertices, original_skin)
    for name, structure in patient.anatomy.structures.items():
        np.testing.assert_array_equal(structure.local_to_world, original_matrices[name])


def test_nonzero_human_origin_is_removed_from_child_poses(tmp_path):
    patient = body()
    root = np.eye(4)
    root[:3, 3] = [0.4, 0.5, 0.6]
    patient.soma.soma_body = PosedBody(
        patient.soma.soma_body.vertices,
        patient.soma.soma_body.faces,
        {},
        {"Root": root},
    )
    stage = Usd.Stage.Open(str(patient.export_to_usd(tmp_path / "origin.usdc")))
    skin = UsdGeom.Mesh(stage.GetPrimAtPath("/HumanBody/Exterior/SOMA"))
    np.testing.assert_allclose(
        skin.GetPointsAttr().Get(),
        patient.soma.soma_body.vertices - root[:3, 3],
        atol=1e-7,
    )
    for prim in stage.GetPrimAtPath("/HumanBody/Anatomy").GetChildren():
        structure = patient.anatomy.structures[prim.GetCustomDataByKey("anatomy:name")]
        np.testing.assert_allclose(
            np.array(UsdGeom.Xformable(prim).GetLocalTransformation()).T,
            np.linalg.inv(root) @ structure.local_to_world,
        )
