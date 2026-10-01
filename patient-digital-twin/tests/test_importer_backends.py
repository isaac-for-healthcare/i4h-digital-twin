# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importer contracts; run pytest tests/test_importer_backends.py (no model downloads)."""

import json
import sys
from pathlib import Path
from types import ModuleType

import nibabel as nib
import numpy as np
import pytest
from patient_digital_twin import AnatomyCollection
from patient_digital_twin.catalog import CATALOG
from patient_digital_twin.importers import (
    NVGenerateImporter,
    NVSegmentImporter,
    SimpleImporter,
)
from patient_digital_twin.importers._common import image_input, segmentation_anatomy


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
    body = segmentation_anatomy(
        nib.Nifti1Image(mask, np.eye(4)), {5: "left kidney", 90: "unrelated"}
    )
    assert isinstance(body, AnatomyCollection)
    assert set(body.structures) == set(CATALOG)
    assert not body.structures["kidney_left"].is_empty
    assert body.structures["colon"].is_empty


@pytest.mark.parametrize("in_process", [True, False])
def test_generation_fresh_seed_and_catalog_import(monkeypatch, tmp_path, in_process):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts/inference.py").write_text("")
    (tmp_path / "configs/label_dict.json").write_text(json.dumps({"colon": 62}))
    import patient_digital_twin.importers.nvgenerate_importer as module

    monkeypatch.setattr(module, "runtime", lambda *args: None if in_process else sys.executable)
    seeds = iter([12, 12, 13])
    monkeypatch.setattr(module.secrets, "randbits", lambda bits: next(seeds))

    def run(command, **kwargs):
        nib.save(
            nib.Nifti1Image(np.ones((3, 3, 3), np.uint8) * 62, np.eye(4)), command[-2]
        )

        nib.save(
            nib.Nifti1Image(np.full((3, 3, 3), 100, dtype=np.float32), np.eye(4)),
            command[-1],
        )

    monkeypatch.setattr(module, "run_backend", run)
    from patient_digital_twin.importers import _nvgenerate_worker

    def generate(seed, output, ct_output):
        assert Path.cwd() == tmp_path
        run([seed, output, ct_output])

    monkeypatch.setattr(_nvgenerate_worker, "generate", generate)
    importer = NVGenerateImporter(source_root=tmp_path)
    with pytest.warns(UserWarning):
        first = importer.to_anatomy_collection()
    assert importer.seed == 12 and not first.structures["colon"].is_empty
    with pytest.warns(UserWarning):
        importer.to_anatomy_collection()
    assert importer.seed == 13
    np.testing.assert_array_equal(importer.ct_volume_zyx, np.full((3, 3, 3), 100))
    assert importer.ct_voxel_to_imaging.shape == (4, 4)


@pytest.mark.parametrize("in_process", [True, False])
def test_nvsegment_requests_supported_prompts_and_uses_output_ids(
    monkeypatch, tmp_path, in_process
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

    monkeypatch.setattr(module, "runtime", lambda *args: None if in_process else sys.executable)

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
    bundle = ModuleType("monai.bundle")

    def bundle_run(*, config_file, meta_file):
        assert Path.cwd() == tmp_path
        assert meta_file == str(tmp_path / "configs/metadata.json")
        run(["--config_file", config_file])

    bundle.run = bundle_run
    monkeypatch.setitem(sys.modules, "monai.bundle", bundle)
    importer = NVSegmentImporter(
        np.zeros((3, 3, 3)),
        bundle_root=tmp_path,
        modality="MR",
        affine_xyz_to_imaging_m=np.diag([0.001, 0.001, 0.001, 1]),
    )
    with pytest.warns(UserWarning):
        body = importer.to_anatomy_collection()
    assert not body.structures["colon"].is_empty and body.structures["liver"].is_empty


def test_simple_default_colon_and_explicit_placement(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    path = tmp_path / "cube.obj"
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(path)
    default = SimpleImporter({"colon": path}).to_anatomy_collection()
    assert default.body_to_imaging is not None
    assert set(default.structures) == {"colon"}
    matrix = np.eye(4)
    matrix[:3, 3] = [0.2, 0.3, 0.4]
    explicit = SimpleImporter(
        {"liver": path}, mesh_to_body={"liver": matrix}
    ).to_anatomy_collection()
    assert explicit.body_to_imaging is None
    np.testing.assert_allclose(
        explicit.structures["liver"].world_vertices.mean(0), matrix[:3, 3]
    )
    matrix[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        SimpleImporter(
            {"liver": path}, mesh_to_body={"liver": matrix}
        ).to_anatomy_collection()


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
    ).to_anatomy_collection()
    np.testing.assert_allclose(
        body.structures["colon"].vertices,
        [[0.1, 0.2, 0.3], [0.11, 0.2, 0.3], [0.1, 0.21, 0.3]],
    )


@pytest.mark.parametrize("fail", [False, True])
def test_paired_worker_keeps_complete_mask_and_matching_image(monkeypatch, tmp_path, fail):
    import runpy

    root = tmp_path / "upstream"
    (root / "configs").mkdir(parents=True)
    (root / "conditions.json").write_text(json.dumps([{"organ_size": [0.5] * 10}]))
    (root / "configs/environment_rflow-ct.json").write_text(
        json.dumps({"all_anatomy_size_conditions_json": "conditions.json"})
    )
    (root / "configs/config_infer.json").write_text(
        json.dumps({"anatomy_list": ["lung tumor"]})
    )
    scripts = ModuleType("scripts")
    scripts.sample = ModuleType("scripts.sample")
    scripts.inference = ModuleType("scripts.inference")
    def original_filter(labels, organs):
        return labels[:1]

    scripts.sample.filter_mask_with_organs = original_filter

    def infer():
        environment = json.loads(Path(sys.argv[sys.argv.index("-e") + 1]).read_text())
        config = json.loads(Path(sys.argv[sys.argv.index("-i") + 1]).read_text())
        assert config["num_output_samples"] == 1
        assert config["controllable_anatomy_size"] == [["liver", 0.5]]
        labels = np.array([1, 2, 62], dtype=np.uint8)
        np.testing.assert_array_equal(
            scripts.sample.filter_mask_with_organs(labels, [1]), labels
        )
        if fail:
            raise RuntimeError("generation failed")
        output = Path(environment["output_dir"])
        output.mkdir()
        nib.save(
            nib.Nifti1Image(np.ones((3, 4, 5), np.uint8), np.eye(4)),
            output / "sample_label.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(np.full((3, 4, 5), 100, np.float32), np.eye(4)),
            output / "sample_image.nii.gz",
        )

    scripts.inference.main = infer
    monkeypatch.setitem(sys.modules, "scripts", scripts)
    mask, ct = tmp_path / "mask.nii.gz", tmp_path / "ct.nii.gz"
    monkeypatch.setattr(sys, "argv", ["worker", "12", str(mask), str(ct)])
    monkeypatch.chdir(root)
    worker = (
        Path(__file__).resolve().parents[1]
        / "patient_digital_twin/importers/_nvgenerate_worker.py"
    )
    original_argv = sys.argv
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            runpy.run_path(str(worker), run_name="__main__")
    else:
        runpy.run_path(str(worker), run_name="__main__")
    assert sys.argv is original_argv
    assert scripts.sample.filter_mask_with_organs is original_filter
    if fail:
        assert not mask.exists() and not ct.exists()
        return
    assert nib.load(mask).shape == nib.load(ct).shape == (3, 4, 5)
    assert np.all(nib.load(ct).get_fdata() == 100)
