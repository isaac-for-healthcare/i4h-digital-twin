# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual USD round trips, world transforms, material bindings and hidden meshes."""

import numpy as np
import pytest
from patient_digital_twin import AnatomicalStructure, HumanBody, Kind  # noqa: E402

pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdShade  # noqa: E402


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
        np.testing.assert_allclose(actual, source, atol=1e-7)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert material
        mesh.MakeInvisible()
    assert not stage.GetPrimAtPath("/HumanBody/Exterior")
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
    patient.attach_imaging(
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
    shape = tuple(ct.GetAttribute("ct:shape").Get())
    np.testing.assert_array_equal(
        np.array(ct.GetAttribute("ct:hu").Get()).reshape(shape), hu.transpose(2, 1, 0)
    )
    matrix = ct.GetAttribute("ct:arrayIndexToScan").Get()
    np.testing.assert_allclose(
        matrix.Transform(Gf.Vec3d(3, 2, 1)), [0.008, 0.014, 0.042]
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
