# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importer contracts; run pytest tests/test_importer_backends.py (no model downloads)."""

import inspect
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import nibabel as nib
import numpy as np
import pytest
from patient_digital_twin import AnatomyCollection
from patient_digital_twin import importers as module
from patient_digital_twin.importers import (
    NVGenerateImporter,
    NVSegmentImporter,
    SimpleImporter,
    segmentation_anatomy,
)


def test_nvsegment_input_is_converted_to_millimeter_nifti():
    affine = np.diag([0.002, 0.003, 0.004, 1])
    affine[:3, 3] = [0.1, -0.2, 0.3]
    image = nib.Nifti1Image(np.zeros((3, 4, 5)), affine)
    image.header.set_xyzt_units("meter")
    converted = NVSegmentImporter(image).image
    expected = affine.copy()
    expected[:3] *= 1000
    np.testing.assert_allclose(converted.affine, expected)
    assert converted.header.get_xyzt_units()[0] == "mm"
    with pytest.raises(ValueError, match="finite 3D"):
        NVSegmentImporter(nib.Nifti1Image(np.full((2, 2, 2), np.nan), np.eye(4)))


def test_segmentation_filters_non_catalog_ids_and_keeps_missing_empty():
    mask = np.zeros((7, 7, 7), np.uint8)
    mask[1:3, 1:3, 1:3] = 5
    mask[4:6, 4:6, 4:6] = 90
    body = segmentation_anatomy(
        nib.Nifti1Image(mask, np.eye(4)), {5: "left kidney", 90: "unrelated", 62: "colon"}
    )
    assert isinstance(body, AnatomyCollection)
    assert set(body.structures) == {"kidney_left", "colon"}
    assert not body.structures["kidney_left"].is_empty
    assert body.structures["colon"].is_empty


@pytest.mark.parametrize("in_process", [True, False])
def test_generation_seed_ct_scan_and_catalog_import(monkeypatch, tmp_path, in_process):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts/inference.py").write_text("")
    (tmp_path / "configs/label_dict.json").write_text(json.dumps({"colon": 62}))
    seeds = iter([12, 13])
    monkeypatch.setattr(module.secrets, "randbits", lambda bits: next(seeds))

    def run(command, **kwargs):
        nib.save(
            nib.Nifti1Image(np.ones((3, 3, 3), np.uint8) * 62, np.eye(4)), command[-2]
        )

        nib.save(
            nib.Nifti1Image(np.full((3, 3, 3), 100, dtype=np.float32), np.eye(4)),
            command[-1],
        )

    monkeypatch.setattr(module.subprocess, "run", run)

    def generate(seed, output, ct_output):
        assert Path.cwd() == tmp_path
        run([seed, output, ct_output])

    monkeypatch.setattr(module, "_generate", generate)
    importer = NVGenerateImporter(source_root=tmp_path, python_executable=None if in_process else sys.executable)
    first = importer.to_anatomy_collection()
    assert importer.seed == 12 and not first.structures["colon"].is_empty
    importer.to_anatomy_collection()
    assert importer.seed == 13
    np.testing.assert_array_equal(importer.ct_scan.values_kji, np.full((3, 3, 3), 100))
    assert importer.ct_scan.metadata["source"]["seed"] == 13


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

    monkeypatch.setattr(module.subprocess, "run", run)
    bundle = ModuleType("monai.bundle")

    def bundle_run(*, config_file, meta_file):
        assert Path.cwd() == tmp_path
        assert meta_file == str(tmp_path / "configs/metadata.json")
        run(["--config_file", config_file])

    bundle.run = bundle_run
    monkeypatch.setitem(sys.modules, "monai.bundle", bundle)
    importer = NVSegmentImporter(
        nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4)),
        bundle_root=tmp_path,
        modality="MR",
        python_executable=None if in_process else sys.executable,
    )
    body = importer.to_anatomy_collection()
    assert not body.structures["colon"].is_empty and "liver" not in body.structures  # CT-only label


def test_simple_default_colon_and_explicit_placement(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    path = tmp_path / "cube.obj"
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(path)
    default = SimpleImporter({"colon": path}).to_anatomy_collection()
    assert default.body_to_imaging is None
    np.testing.assert_array_equal(default.structures["colon"].local_to_body, np.eye(4))
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


def test_simple_rejects_unsupported_formats(tmp_path):
    pytest.importorskip("trimesh")
    with pytest.raises(ValueError, match="Supported mesh formats"):
        SimpleImporter({"colon": tmp_path / "mesh.usda"}).to_anatomy_collection()


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("data_dir_set", [True, False])
def test_paired_worker_keeps_complete_mask_and_matching_image(monkeypatch, tmp_path, fail, data_dir_set):
    root = tmp_path / "upstream"
    (root / "configs").mkdir(parents=True)
    # As upstream: dataset paths are relative to MONAI_DATA_DIRECTORY, not the checkout.
    (root / "configs/environment_rflow-ct.json").write_text(
        json.dumps({"all_anatomy_size_conditions_json": "datasets/all_anatomy_size_conditions.json"})
    )
    (root / "configs/config_infer.json").write_text(
        json.dumps({"anatomy_list": ["lung tumor"]})
    )
    data_dir = tmp_path / "monai_data"
    if data_dir_set:
        (data_dir / "datasets").mkdir(parents=True)
        (data_dir / "datasets/all_anatomy_size_conditions.json").write_text(json.dumps([{"organ_size": [0.5] * 10}]))
        monkeypatch.setenv("MONAI_DATA_DIRECTORY", str(data_dir))
    else:
        monkeypatch.delenv("MONAI_DATA_DIRECTORY", raising=False)
    downloads = []

    def download_model_data(version, root_dir):
        """Stub upstream download: provides the conditions file only under the data root."""
        downloads.append((version, root_dir))
        target = Path(root_dir) / "datasets/all_anatomy_size_conditions.json"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps([{"organ_size": [0.5] * 10}]))

    scripts = ModuleType("scripts")
    scripts.sample = ModuleType("scripts.sample")
    scripts.inference = ModuleType("scripts.inference")
    downloader = ModuleType("scripts.download_model_data")
    downloader.download_model_data = download_model_data
    monkeypatch.setitem(sys.modules, "scripts.download_model_data", downloader)
    def original_filter(labels, organs):
        return labels[:1]

    scripts.sample.filter_mask_with_organs = original_filter

    def infer():
        environment = json.loads(Path(sys.argv[sys.argv.index("-e") + 1]).read_text())
        config = json.loads(Path(sys.argv[sys.argv.index("-i") + 1]).read_text())
        data_root = os.environ["MONAI_DATA_DIRECTORY"]  # Inference resolves the same root.
        assert downloads == [("rflow-ct", data_root)]
        assert environment["all_anatomy_size_conditions_json"] == os.path.join(
            data_root, "datasets/all_anatomy_size_conditions.json"
        )
        assert data_root == str(data_dir) if data_dir_set else data_root.startswith(str(tmp_path))
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
    original_argv = sys.argv
    # The separate-process path runs this exact source with `python -c`.
    code = inspect.getsource(module._generate) + "\nimport sys\n_generate(*sys.argv[1:])\n"
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            exec(compile(code, "<worker>", "exec"), {"__name__": "__main__"})
    else:
        exec(compile(code, "<worker>", "exec"), {"__name__": "__main__"})
    assert sys.argv is original_argv
    assert scripts.sample.filter_mask_with_organs is original_filter
    assert os.environ.get("MONAI_DATA_DIRECTORY") == (str(data_dir) if data_dir_set else None)
    if fail:
        assert not mask.exists() and not ct.exists()
        return
    assert nib.load(mask).shape == nib.load(ct).shape == (3, 4, 5)
    assert np.all(nib.load(ct).get_fdata() == 100)
