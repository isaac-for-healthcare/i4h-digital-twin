# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importer contracts; run pytest examples/test_importers.py (no model downloads)."""

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType

import nibabel as nib
import numpy as np
import pytest
from patient_digital_twin import HumanBody
from patient_digital_twin.catalog import CATALOG
from patient_digital_twin.importers import (
    NVGenerateImporter,
    NVSegmentImporter,
    SimpleImporter,
    TotalSegmentatorImporter,
)
from patient_digital_twin.importers._common import image_input, segmentation_body


def test_array_coordinates_require_physical_affine():
    values = np.zeros((3, 4, 5))
    with pytest.raises(ValueError, match="affine"):
        image_input(values)
    affine = np.diag([0.002, 0.003, 0.004, 1])
    affine[:3, 3] = [0.1, -0.2, 0.3]
    image = image_input(values, affine)
    expected = affine.copy()
    expected[:3] *= 1000
    np.testing.assert_allclose(image.affine, expected)
    assert image.header.get_xyzt_units()[0] == "mm"


def test_segmentation_filters_non_catalog_ids_and_keeps_missing_empty():
    mask = np.zeros((7, 7, 7), np.uint8)
    mask[1:3, 1:3, 1:3] = 5
    mask[4:6, 4:6, 4:6] = 90
    body = segmentation_body(
        nib.Nifti1Image(mask, np.eye(4)), {5: "left kidney", 90: "unrelated"}
    )
    assert isinstance(body, HumanBody)
    assert set(body.structures) == set(CATALOG)
    assert not body.structures["kidney_left"].is_empty
    assert body.structures["colon"].is_empty


def test_total_api_receives_modality_catalog_subset_and_preserves_ids(monkeypatch):
    api, maps = (
        ModuleType("totalsegmentator.python_api"),
        ModuleType("totalsegmentator.map_to_binary"),
    )
    maps.class_map = {
        "total": {5: "liver", 6: "unrelated"},
        "total_mr": {9: "kidney_left"},
    }
    calls = []

    def run(image, **options):
        calls.append(options)
        return nib.Nifti1Image(np.full(image.shape, 9, np.uint8), image.affine)

    api.totalsegmentator = run
    monkeypatch.setitem(sys.modules, "totalsegmentator", ModuleType("totalsegmentator"))
    monkeypatch.setitem(sys.modules, api.__name__, api)
    monkeypatch.setitem(sys.modules, maps.__name__, maps)
    importer = TotalSegmentatorImporter(
        np.zeros((3, 3, 3)), modality="MR", affine_xyz_to_imaging_m=np.eye(4)
    )
    with pytest.warns(UserWarning):
        body = importer.to_human_body()
    assert calls[0]["task"] == "total_mr" and calls[0]["roi_subset"] == ["kidney_left"]
    assert not body.structures["kidney_left"].is_empty
    assert "liver" in importer.report["unsupported"]


def test_optional_total_dependency_fails_with_install_hint(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("totalsegmentator"):
            raise ImportError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="pip install TotalSegmentator"):
        TotalSegmentatorImporter(np.zeros((2, 2, 2))).to_human_body()


def test_generation_fresh_seed_and_catalog_import(monkeypatch, tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts/sample_mask.py").write_text("")
    (tmp_path / "configs/label_dict.json").write_text(json.dumps({"colon": 62}))
    import patient_digital_twin.importers.nvgenerate_importer as module

    monkeypatch.setattr(module, "runtime", lambda *args: sys.executable)
    seeds = iter([12, 12, 13])
    monkeypatch.setattr(module.secrets, "randbits", lambda bits: next(seeds))

    def run(command, **kwargs):
        nib.save(
            nib.Nifti1Image(np.ones((3, 3, 3), np.uint8) * 62, np.eye(4)), command[-1]
        )

    monkeypatch.setattr(module, "run_backend", run)
    importer = NVGenerateImporter(source_root=tmp_path)
    with pytest.warns(UserWarning):
        first = importer.to_human_body()
    assert importer.seed == 12 and not first.structures["colon"].is_empty
    with pytest.warns(UserWarning):
        importer.to_human_body()
    assert importer.seed == 13


def test_nvsegment_requests_supported_prompts_and_uses_output_ids(
    monkeypatch, tmp_path
):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/inference.json").write_text("{}")
    (tmp_path / "configs/label_dict.json").write_text(
        json.dumps(
            {
                "liver": {"index": 1, "datasets": ["CT"]},
                "colon": {"index": 62, "datasets": ["CT", "MRI"]},
                "unrelated": {"index": 90, "datasets": ["MRI"]},
            }
        )
    )
    import patient_digital_twin.importers.nvsegment_importer as module

    monkeypatch.setattr(module, "runtime", lambda *args: sys.executable)

    def run(command, **kwargs):
        config = json.loads(
            Path(command[command.index("--config_file") + 1]).read_text()
        )
        assert config["everything_labels"] == [62]
        assert config["modality"] == "MRI_BODY"
        out = Path(config["output_dir"])
        out.mkdir()
        image = nib.load(config["input_dict"]["image"])
        nib.save(
            nib.Nifti1Image(np.full(image.shape, 62, np.uint8), image.affine),
            out / "image_segmentation.nii.gz",
        )

    monkeypatch.setattr(module, "run_backend", run)
    importer = NVSegmentImporter(
        np.zeros((3, 3, 3)),
        bundle_root=tmp_path,
        modality="MR",
        affine_xyz_to_imaging_m=np.diag([0.001, 0.001, 0.001, 1]),
    )
    with pytest.warns(UserWarning):
        body = importer.to_human_body()
    assert not body.structures["colon"].is_empty and body.structures["liver"].is_empty


def test_simple_default_colon_and_explicit_placement(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    default = SimpleImporter(
        {"colon": Path(__file__).parent / "data/colon.stl"}
    ).to_human_body()
    assert default.body_to_imaging is not None and default.landmarks
    assert set(default.structures) == {"colon"}
    path = tmp_path / "cube.obj"
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(path)
    matrix = np.eye(4)
    matrix[:3, 3] = [0.2, 0.3, 0.4]
    explicit = SimpleImporter(
        {"liver": path}, mesh_to_body={"liver": matrix}
    ).to_human_body()
    assert explicit.body_to_imaging is None and explicit.landmarks == {}
    np.testing.assert_allclose(
        explicit.structures["liver"].world_vertices.mean(0), matrix[:3, 3]
    )
    matrix[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        SimpleImporter({"liver": path}, mesh_to_body={"liver": matrix}).to_human_body()


def test_usd_units_authored_transform_and_winding(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    path = tmp_path / "mesh.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    mesh = UsdGeom.Mesh.Define(stage, "/colon")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
    stage.GetRootLayer().Save()
    body = SimpleImporter(
        {"colon": path}, mesh_to_body={"colon": np.eye(4)}
    ).to_human_body()
    np.testing.assert_allclose(
        body.structures["colon"].vertices,
        [[0.1, 0.2, 0.3], [0.11, 0.2, 0.3], [0.1, 0.21, 0.3]],
    )
