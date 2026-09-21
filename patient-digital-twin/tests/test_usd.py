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
        structure = patient.structures[name]
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
        np.testing.assert_allclose(actual, source[:, [0, 2, 1]] * [1, -1, 1], atol=1e-7)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        assert material
        mesh.MakeInvisible()
    skin = UsdGeom.Mesh(stage.GetPrimAtPath("/HumanBody/Exterior/SOMA"))
    assert skin.ComputeVisibility() == "inherited"
    assert skin.GetDisplayOpacityAttr().Get()[0] == pytest.approx(0.15)
    assert not stage.GetRootLayer().GetExternalReferences()
    assert patient.structures["left organ"].enabled


def test_bad_mesh_does_not_overwrite_existing_file(tmp_path):
    target = tmp_path / "existing.usda"
    target.write_text("original")
    patient = body()
    patient.structures["left organ"].faces = np.array([[0, 1, 999]])
    with pytest.raises(ValueError, match="Invalid triangle mesh"):
        patient.export_to_usd(target)
    assert target.read_text() == "original"
    with pytest.raises(ValueError, match="Attach SOMA"):
        HumanBody().export_to_usd(target)
