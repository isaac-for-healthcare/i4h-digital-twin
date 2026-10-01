# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import nibabel as nib
import numpy as np
import pytest
from patient_digital_twin import __main__ as main
from patient_digital_twin.importers._common import segmentation_anatomy, selected_labels


def test_named_selection_uses_native_ids_and_limits_meshes():
    image = nib.Nifti1Image(np.full((5, 5, 8), 6, np.uint8), np.eye(4))
    anatomy = segmentation_anatomy(image, {6: "aorta", 12: "liver"}, names=["aorta"])
    assert set(anatomy.structures) == {"aorta"}
    assert not anatomy.structures["aorta"].is_empty
    assert selected_labels({6: "aorta", 12: "liver"}, ["aorta"]) == {6: "aorta"}
    with pytest.raises(ValueError, match="Unsupported"):
        selected_labels({12: "liver"}, ["aorta"])


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, "requires --input"),
        ({"classes": ["not_anatomy"]}, "Unknown"),
        ({"classes": []}, "at least one"),
        ({"source": "nvgenerate", "input": "ct.nii.gz"}, "omit --input"),
        ({"source": "nvgenerate", "modality": "MR"}, "CT only"),
    ],
)
def test_invalid_arguments_fail_before_model(options, message, tmp_path):
    args = {
        "source": "nvsegment",
        "classes": ["aorta"],
        "output": tmp_path / "patient.usdc",
    }
    args.update(options)
    with pytest.raises(ValueError, match=message):
        main.run_pipeline(**args)


@pytest.mark.parametrize("modality", ["CT", "MR"])
def test_nvsegment_to_real_usd(monkeypatch, tmp_path, modality):
    pytest.importorskip("pxr")
    from pxr import Usd

    ct = tmp_path / "ct.nii.gz"
    image = nib.Nifti1Image(np.full((9, 9, 16), 100, np.float32), np.eye(4))
    nib.save(image, ct)

    def segment(self, *, names):
        assert names == ("liver",)
        return segmentation_anatomy(
            nib.Nifti1Image(np.ones(image.shape, np.uint8), image.affine),
            {1: "liver"},
            names=names,
        )

    monkeypatch.setattr(main.NVSegmentImporter, "to_anatomy_collection", segment)
    output = main.run_pipeline(
        source="nvsegment",
        input=ct,
        classes=["liver"],
        output=tmp_path / "patient.usdc",
        modality=modality,
    )
    stage = Usd.Stage.Open(str(output))
    names = [p.GetCustomDataByKey("anatomy:name") for p in stage.Traverse()]
    assert "liver" in names and "aorta" not in names
    assert bool(stage.GetPrimAtPath("/HumanBody/Imaging/CT")) == (modality == "CT")


@pytest.mark.parametrize("stored", [False, True])
def test_workflow_centerline_uses_ct_grid_and_preserves_structure_graph(
    monkeypatch, tmp_path, stored
):
    pytest.importorskip("pxr")
    pytest.importorskip("vtk")
    import yaml
    from patient_digital_twin import HumanBody

    shape = (25, 25, 41)
    x, y, z = np.indices(shape)
    mask = (((x - 12) ** 2 + (y - 12) ** 2 < 25) & (z > 3) & (z < 37)).astype(np.uint8)
    image = nib.Nifti1Image(
        np.where(mask, 300, 40).astype(np.float32), np.diag([1.0, 1.0, 2.0, 1.0])
    )
    ct = tmp_path / "ct.nii.gz"
    nib.save(image, ct)
    anatomy = segmentation_anatomy(
        nib.Nifti1Image(mask, image.affine), {1: "aorta"}, names=["aorta"]
    )
    if stored:
        HumanBody(anatomy).extract_topology(spacing_m=0.0015)
    previous = anatomy.structures["aorta"].centerline
    monkeypatch.setattr(
        main.NVSegmentImporter, "to_anatomy_collection", lambda self, **kw: anatomy
    )
    args = main.parser().parse_args(
        [
            "--source",
            "nvsegment",
            "--input",
            str(ct),
            "--classes",
            "aorta",
            "--output",
            str(tmp_path / "bundle"),
            "--format",
            "workflow",
        ]
    )
    path = main.run_pipeline(**vars(args))
    hu = np.load(path.parent / "hu_volume.npy")
    expected_mu = np.interp(hu, [-1000, 3000], [0, 0.02]).astype(np.float32)
    np.testing.assert_array_equal(np.load(path.parent / "mu_volume.npy"), expected_mu)
    manifest = yaml.safe_load(path.read_text())
    for relative in manifest["artifacts"].values():
        assert (path.parent / relative).is_file()
    graph = anatomy.structures["aorta"].centerline
    if stored:
        assert graph is previous
    assert len(graph.points) > 2
    points = np.load(path.parent / "centerline_points_mm.npy")
    np.testing.assert_allclose(points[:, :2], -12.0, atol=2)
    assert points[:, 2].min() > 6 and points[:, 2].max() < 74
    assert np.load(path.parent / "centerline_radii_mm.npy").min() > 0
    assert manifest["coordinate_frame"] == "DICOM_LPS"
    assert "aorta" in manifest["centerlines"]

    # Navigation uses the final CT-grid mask, even when local graphs exist.
    from scipy.ndimage import distance_transform_edt

    metadata = json.loads((path.parent / "metadata.json").read_text())
    spacing = np.asarray(metadata["spacing_zyx_mm"])
    origin = np.asarray(metadata["origin_xyz_mm"])
    voxel_xyz = (points - origin) / spacing[::-1]
    np.testing.assert_allclose(voxel_xyz, np.rint(voxel_xyz), atol=1e-6)
    indices = np.rint(voxel_xyz).astype(int)[:, ::-1]
    final_mask = np.load(path.parent / "vessel_mask.npy")
    assert final_mask[tuple(indices.T)].all()
    distances = distance_transform_edt(final_mask, sampling=spacing)
    np.testing.assert_array_equal(
        np.load(path.parent / "centerline_radii_mm.npy"),
        distances[tuple(indices.T)].astype(np.float32),
    )
    edges = np.load(path.parent / "centerline_edges.npy")
    assert len(edges) == len(points) - 1  # One unbranched tube.
    neighbor_offsets = np.abs(indices[edges[:, 1]] - indices[edges[:, 0]])
    assert (neighbor_offsets.max(axis=1) == 1).all()
    with np.load(path.parent / manifest["centerlines"]["aorta"]["path"]) as saved:
        for key in ("points", "edges", "radii"):
            np.testing.assert_array_equal(saved[key], getattr(graph, key))
