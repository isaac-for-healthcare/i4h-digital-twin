# SPDX-License-Identifier: Apache-2.0
"""Simulation exports retain source coordinates and usable surface topology."""

import numpy as np
import pytest
import yaml

pytest.importorskip("pxr")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("shapely")
from patient_digital_twin.exporters.physics_export import (
    _mesh,
    _xcath,
    export_physics_examples,
)
from pxr import Gf, Usd, UsdGeom


def test_patient_mesh_bakes_hierarchy_and_units(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    parent = UsdGeom.Xform.Define(stage, "/Patient")
    parent.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
    original = trimesh.creation.box()
    mesh = UsdGeom.Mesh.Define(stage, "/Patient/aorta")
    mesh.CreatePointsAttr(original.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(original.faces))
    mesh.CreateFaceVertexIndicesAttr(original.faces.ravel().tolist())
    result = _mesh(stage, "/Patient/aorta")
    np.testing.assert_allclose(result.bounds, original.bounds * 0.01 + [0.1, 0.2, 0.3])
    mesh.CreateVisibilityAttr("invisible")
    with pytest.raises(ValueError, match="disabled"):
        _mesh(stage, "/Patient/aorta")


def test_xcath_entry_and_units_are_recorded(tmp_path):
    mesh = trimesh.creation.cylinder(radius=0.012, height=0.25, sections=32)
    template = {
        "catheters": [{"centerline": {"point_count": 128, "segment_length": 0.05}}],
        "vessels": [
            {
                "source": {"path": "stock.usda"},
                "display": {},
                "centerline": {"path": "stock.json"},
            }
        ],
    }
    result = _xcath(mesh, template, tmp_path, "aorta")
    scene = yaml.safe_load((tmp_path / "aorta.yaml").read_text())
    assert scene["vessels"][0]["source"]["path"] == "aorta.usda"
    assert "centerline" not in scene["vessels"][0]
    assert template["vessels"][0]["source"]["path"] == "stock.usda"
    matrix = np.array(result["scene_from_patient_m"])
    entry = np.r_[result["entry_patient_m"], 1]
    np.testing.assert_allclose(matrix @ entry, [6.25, 0, 1, 1])
    stage = Usd.Stage.Open(str(tmp_path / "aorta.usda"))
    actual = _mesh(stage, "/Anatomy")
    _, count = np.unique(actual.edges_sorted, axis=0, return_counts=True)
    assert count.max() == 2 and (count == 1).any()
    assert actual.area_faces.min() > 0


def test_missing_anatomy_does_not_create_output(tmp_path):
    source = tmp_path / "patient_twin.yaml"
    source.write_text(yaml.safe_dump({"anatomy": {"structures": {}}}))
    target = tmp_path / "export"
    with pytest.raises(ValueError, match="aorta"):
        export_physics_examples(source, target, physics_root=tmp_path)
    assert not target.exists()
