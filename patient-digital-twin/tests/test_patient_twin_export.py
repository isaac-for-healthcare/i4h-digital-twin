# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CT and anatomy share the exported workflow coordinate chain."""

from patient_digital_twin import HumanBody
import json

import nibabel as nib
import numpy as np
import pytest
import yaml
from patient_digital_twin import SegmentationImporter

pytest.importorskip("vtk")
pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom


@pytest.mark.parametrize("with_soma", [False, True])
def test_bundle_uses_original_geometry_and_consistent_lps(tmp_path, with_soma):
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
    if with_soma:
        from test_soma import ArticulatedLayer

        registration = np.eye(4)
        registration[:3, 3] = [0.3, 0.2, 0.1]
        body.AttachExternalBody(ArticulatedLayer(), body_to_soma=registration)
        scan_skin = body.soma.soma_body.vertices.copy()
        body.soma.pose(transl=[[10, 20, 30]])
        displayed_skin = body.soma.soma_body.vertices.copy()
    body.anatomy.structures["aorta"].local_to_world[:3, 3] += (
        10  # display pose must not move source CT anatomy
    )
    original = body.anatomy.structures["aorta"].local_to_world.copy()
    target = body.export_patient_twin(tmp_path / "bundle", vessel_names=["aorta"], exterior="auto" if with_soma else "ct")
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
    if with_soma:
        from patient_digital_twin.geometry import transform_points

        skin = UsdGeom.Mesh(stage.GetPrimAtPath("/HumanBody/Exterior/SOMA"))
        expected = transform_points(
            scan_skin,
            np.diag([-1, -1, 1, 1])
            @ body.imaging.body_to_imaging
            @ np.linalg.inv(registration),
        )
        np.testing.assert_allclose(skin.GetPointsAttr().Get(), expected, atol=1e-7)
        np.testing.assert_array_equal(body.soma.soma_body.vertices, displayed_skin)
        assert manifest["anatomy"]["exterior"]["source"] == "SOMA"
        assert not stage.GetPrimAtPath("/HumanBody/Exterior/CT")
    else:
        assert stage.GetPrimAtPath("/HumanBody/Exterior/CT")
    with pytest.raises(FileExistsError):
        body.export_patient_twin(target.parent, vessel_names=["aorta"])
