# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CT and anatomy share the exported workflow coordinate chain."""

import json

import nibabel as nib
import numpy as np
import pytest
import yaml
from patient_digital_twin import HumanBody, SegmentationImporter

pytest.importorskip("vtk")
pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom


def test_bundle_uses_original_geometry_and_consistent_lps(tmp_path):
    shape = (30, 24, 24)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[3:27, 9:14, 9:14] = 1
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = [30, 40, 0]
    hu = np.full(shape, -1000.0, dtype=np.float32)
    hu[2:-2, 2:-2, 2:-2] = 100
    image = nib.Nifti1Image(hu.transpose(2, 1, 0), affine)
    image.header.set_xyzt_units("mm")
    ct_path = tmp_path / "ct.nii.gz"
    nib.save(image, ct_path)
    affine_m = affine.copy()
    affine_m[:3] *= 0.001
    body = HumanBody(
        SegmentationImporter.from_array(
            mask, {1: "aorta"}, affine_xyz_to_imaging_m=affine_m
        ).to_anatomy_collection()
    )
    body.AttachImaging(hu, voxel_to_imaging=affine_m, source_path=ct_path)
    body.anatomy.structures["aorta"].local_to_world[:3, 3] += (
        10  # display pose must not move source CT anatomy
    )
    original = body.anatomy.structures["aorta"].local_to_world.copy()
    target = body.export_patient_twin(
        tmp_path / "bundle", vessel_names=["aorta"], exterior="ct"
    )
    manifest = yaml.safe_load(target.read_text())
    assert manifest["coordinate_frame"] == "DICOM_LPS"
    for value in manifest["artifacts"].values():
        assert (target.parent / value).exists()
    metadata = json.loads((target.parent / "metadata.json").read_text())
    assert tuple(metadata["shape_zyx"]) == shape
    np.testing.assert_array_equal(np.load(target.parent / "vessel_mask.npy"), mask)
    points = np.load(target.parent / "centerline_points_mm.npy")
    voxel = points - np.array([-30, -40, 0])
    indices = np.rint(voxel).astype(int)[:, ::-1]
    assert mask[tuple(indices.T)].all()
    stage = Usd.Stage.Open(str(target.parent / "patient_anatomy.usdc"))
    prim = stage.GetPrimAtPath("/HumanBody/Anatomy/aorta")
    xf = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    actual = np.array(
        [xf.Transform(Gf.Vec3d(p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
    )
    np.testing.assert_allclose(
        actual, body.imaging_vertices("aorta") * [-1, -1, 1], atol=1e-7
    )
    np.testing.assert_array_equal(
        body.anatomy.structures["aorta"].local_to_world, original
    )
    assert manifest["anatomy"]["structures"]["aorta"]["prim_path"] == str(
        prim.GetPath()
    )
    skin = UsdGeom.Mesh(stage.GetPrimAtPath("/HumanBody/Exterior/CT"))
    assert skin and len(skin.GetPointsAttr().Get())
    assert skin.GetDisplayOpacityAttr().Get()[0] == pytest.approx(0.15)
    assert manifest["anatomy"]["exterior"]["source"] == "CT"
    assert manifest["anatomy"]["exterior"]["pose"] == "imaging"
    assert not stage.GetRootLayer().GetExternalReferences()
    with pytest.raises(FileExistsError):
        body.export_patient_twin(target.parent, vessel_names=["aorta"])
