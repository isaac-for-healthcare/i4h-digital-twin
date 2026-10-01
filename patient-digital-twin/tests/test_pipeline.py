# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline branch contracts and real mesh/topology bundle exports."""

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml
from patient_digital_twin import AnatomicalStructure, AnatomyCollection, HumanBody, Kind


@pytest.fixture
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "pipeline", Path(__file__).parents[1] / "examples/pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "source", ["nvgenerate", "nvsegment", "simple"]
)
def test_input_branches_attach_only_matching_ct(
    pipeline, monkeypatch, tmp_path, source
):
    anatomy = AnatomyCollection()
    anatomy.body_to_imaging = np.eye(4)
    calls = []

    class Importer:
        ct_volume_zyx = np.full((3, 4, 5), 321, np.float32)
        ct_voxel_to_imaging = np.diag([0.001, 0.002, 0.003, 1])

        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def to_anatomy_collection(self):
            return anatomy

    for name in (
        "NVGenerateImporter",
        "NVSegmentImporter",
        "SimpleImporter",
    ):
        monkeypatch.setattr(pipeline, name, Importer)
    input_path = tmp_path / "input.nii.gz"
    nib.save(
        nib.Nifti1Image(np.full((5, 4, 3), 123, np.float32), np.diag([1, 2, 3, 1])),
        input_path,
    )
    if source == "simple":
        input_path = tmp_path / "meshes.json"
        input_path.write_text(json.dumps({"meshes": {"liver": "liver.stl"}}))
    options = ["--source", source, "--output", str(tmp_path / "out")]
    if source != "nvgenerate":
        options += ["--input", str(input_path)]
    body = pipeline.import_body(pipeline.parser().parse_args(options))
    assert body.anatomy is anatomy
    if source == "simple":
        assert body.imaging is None
        np.testing.assert_array_equal(calls[0][1]["mesh_to_body"]["liver"], np.eye(4))
    else:
        assert body.imaging.volume.shape == (3, 4, 5)
        assert np.all(body.imaging.volume == (321 if source == "nvgenerate" else 123))
        np.testing.assert_allclose(
            body.imaging.voxel_to_imaging, Importer.ct_voxel_to_imaging
        )


@pytest.mark.parametrize("with_ct", [False, True])
def test_pipeline_order_and_complete_exports(pipeline, monkeypatch, tmp_path, with_ct):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("pxr")
    from pxr import Usd

    mesh = trimesh.creation.cylinder(radius=0.005, height=0.04, sections=16)
    anatomy = AnatomyCollection(
        {
            "aorta": AnatomicalStructure(
                "aorta", Kind.VESSEL, mesh.vertices.copy(), mesh.faces.copy()
            )
        }
    )
    events = []

    class Body(HumanBody):
        def extract_topology(self, **options):
            events.append("topology")
            return super().extract_topology(**options)

        def export_patient_twin(self, *args, **kwargs):
            events.append("bundle")
            return super().export_patient_twin(*args, **kwargs)

        def export_to_usd(self, *args, **kwargs):
            events.append("usd")
            return super().export_to_usd(*args, **kwargs)

    body = Body(anatomy)
    if with_ct:
        body.AttachImaging(
            np.full((8, 8, 8), 100, np.float32),
            voxel_to_imaging=np.diag([0.001] * 3 + [1]),
            body_to_imaging=np.eye(4),
        )
    monkeypatch.setattr(pipeline, "import_body", lambda args: body)
    options = [
        "--source",
        "nvgenerate" if with_ct else "simple",
        "--output",
        str(tmp_path / "out"),
    ]
    if not with_ct:
        options += ["--input", "meshes.json"]
    manifest_path = pipeline.run(pipeline.parser().parse_args(options))
    assert events == ["topology", "bundle", "usd"]
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["anatomy"]["exterior"] is None
    assert manifest["coordinate_frame"] == ("DICOM_LPS" if with_ct else "body")
    assert ("hu_volume" in manifest["artifacts"]) == with_ct
    graph = body.anatomy.structures["aorta"].centerline
    with np.load(
        manifest_path.parent / manifest["centerlines"]["aorta"]["path"]
    ) as saved:
        np.testing.assert_array_equal(saved["points"], graph.points)
    for filename in ("human_body.usdc", "patient_anatomy.usdc"):
        stage = Usd.Stage.Open(str(manifest_path.parent / filename))
        prim = stage.GetPrimAtPath("/HumanBody/Anatomy/aorta")
        np.testing.assert_allclose(
            prim.GetAttribute("centerline:points").Get(), graph.points, atol=1e-7
        )
        assert not stage.GetPrimAtPath("/HumanBody/Exterior")


def test_pipeline_failure_leaves_no_partial_bundle(pipeline, monkeypatch, tmp_path):
    from unittest.mock import Mock

    body = HumanBody(
        {
            "liver": AnatomicalStructure(
                "liver", Kind.ORGAN, np.zeros((3, 3)), np.array([[0, 1, 2]])
            )
        }
    )
    monkeypatch.setattr(pipeline, "import_body", lambda args: body)
    monkeypatch.setattr(
        body, "export_patient_twin", Mock(side_effect=RuntimeError("export failed"))
    )
    output = tmp_path / "out"
    args = pipeline.parser().parse_args(
        ["--source", "simple", "--input", "meshes.json", "--output", str(output)]
    )
    with pytest.raises(RuntimeError, match="export failed"):
        pipeline.run(args)
    assert not output.exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--anatomy", "typo"],
        ["--source", "simple", "--input", "meshes.json", "--ct", "ct.nii"],
        ["--source", "nvsegment"],
    ],
)
def test_invalid_options_fail_before_inference(
    pipeline, monkeypatch, tmp_path, options
):
    monkeypatch.setattr(
        pipeline, "import_body", lambda args: pytest.fail("Inference must not start")
    )
    with pytest.raises(ValueError):
        pipeline.run(
            pipeline.parser().parse_args(["--output", str(tmp_path / "out"), *options])
        )


def test_simple_stl_pipeline_without_imaging(pipeline, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    mesh = trimesh.creation.box(extents=[0.02, 0.03, 0.04])
    mesh.apply_translation([0.2, 0.3, 0.4])
    mesh.export(tmp_path / "liver.stl")
    config = tmp_path / "meshes.json"
    config.write_text(json.dumps({"meshes": {"liver": "liver.stl"}}))
    target = pipeline.run(
        pipeline.parser().parse_args(
            [
                "--source",
                "simple",
                "--input",
                str(config),
                "--output",
                str(tmp_path / "out"),
            ]
        )
    )
    manifest = yaml.safe_load(target.read_text())
    assert manifest["coordinate_frame"] == "body"
    np.testing.assert_array_equal(
        manifest["transforms"]["world_from_patient_m"], np.eye(4)
    )
    assert manifest["anatomy"]["exterior"] is None
    assert set(manifest["artifacts"]) == {"anatomy_usd"}
    stage = Usd.Stage.Open(str(target.parent / "human_body.usdc"))
    points = np.asarray(
        UsdGeom.Mesh(stage.GetPrimAtPath("/HumanBody/Anatomy/liver"))
        .GetPointsAttr()
        .Get()
    )
    np.testing.assert_allclose(points.min(0), mesh.vertices.min(0), atol=1e-7)
    np.testing.assert_allclose(points.max(0), mesh.vertices.max(0), atol=1e-7)
