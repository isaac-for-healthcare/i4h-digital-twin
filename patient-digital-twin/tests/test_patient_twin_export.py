# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CT and anatomy share the exported workflow coordinate chain."""

import nibabel as nib
import numpy as np
import pytest
import yaml
from patient_digital_twin import HumanBody, SegmentationImporter  # noqa: E402
from patient_digital_twin.geometry import transform_points  # noqa: E402

pytest.importorskip("vtk")
pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom  # noqa: E402


@pytest.mark.parametrize("angle", [0.0, 0.35])
def test_bundle_preserves_native_grid_and_oblique_scan_geometry(tmp_path, angle):
    shape = (30, 24, 24)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[3:27, 9:14, 9:14] = 1
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    rotation = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    affine[:3, :3] = rotation @ affine[:3, :3]
    affine[:3, 3] = [30, 40, 0]
    hu = np.full(shape, -1000.0, dtype=np.float32)
    hu[2:-2, 2:-2, 2:-2] = 100
    image = nib.Nifti1Image(hu.transpose(2, 1, 0), affine)
    image.header.set_xyzt_units("mm")
    ct_path = tmp_path / "ct.nii.gz"
    nib.save(image, ct_path)
    affine = nib.load(ct_path).affine
    affine_m = affine.copy()
    affine_m[:3] *= 0.001
    body = HumanBody(
        SegmentationImporter(
            mask, {1: "aorta"}, affine_xyz_to_imaging_m=affine_m
        ).to_anatomy_collection()
    )
    from patient_digital_twin.scan_volume import from_nifti

    body.attach_scan(from_nifti(ct_path), source_path=ct_path)
    body.anatomy.structures["aorta"].local_to_world[:3, 3] += (
        10  # display pose must not move source CT anatomy
    )
    original = body.anatomy.structures["aorta"].local_to_world.copy()
    target = body.export_patient_twin(
        tmp_path / "bundle", vessel_names=["aorta"], ct_exterior=True
    )
    manifest = yaml.safe_load(target.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["coordinate_frame"] == "RAS"
    for value in manifest["artifacts"].values():
        assert (target.parent / value).exists()
    metadata = yaml.safe_load((target.parent / "volume.yaml").read_text())["output"]
    assert tuple(metadata["shape"]) == shape[::-1]
    np.testing.assert_array_equal(
        np.load(target.parent / "volume.npy"), hu.transpose(2, 1, 0)
    )
    np.testing.assert_array_equal(
        np.load(target.parent / "vessel_mask.npy"), mask.transpose(2, 1, 0)
    )
    points = np.load(target.parent / "centerline_points.npy")
    inverse = np.linalg.inv(affine)
    voxel = points @ inverse[:3, :3].T + inverse[:3, 3]
    indices = np.rint(voxel).astype(int)[:, ::-1]
    assert mask[tuple(indices.T)].all()
    assert "world_from_patient_m" not in manifest["transforms"]
    stage = Usd.Stage.Open(str(target.parent / "patient_anatomy.usdc"))
    prim = stage.GetPrimAtPath("/HumanBody/Anatomy/aorta")
    xf = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    actual = np.array(
        [xf.Transform(Gf.Vec3d(p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
    )
    np.testing.assert_allclose(actual, transform_points(body.anatomy.structures["aorta"].body_vertices, body.imaging.body_to_imaging) * 1000, atol=1e-5)
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


def test_ct_exterior_is_one_body_surface(tmp_path):
    from patient_digital_twin.scan_volume import from_nifti
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    shape = (42, 42, 42)
    hu = np.full(shape, -1000.0, dtype=np.float32)
    hu[6:36, 6:30, 6:36] = 40  # body
    hu[15:24, 12:21, 15:24] = -900  # enclosed gas pocket
    hu[6:36, 33:39, 6:36] = 400  # table, separate from the body
    mask = np.zeros(shape, dtype=np.uint8)
    mask[9:33, 15:18, 27:30] = 1
    image = nib.Nifti1Image(hu.transpose(2, 1, 0), np.eye(4))
    image.header.set_xyzt_units("mm")
    nib.save(image, tmp_path / "ct.nii.gz")
    body = HumanBody(SegmentationImporter(mask, {1: "aorta"}, affine_xyz_to_imaging_m=np.diag([1e-3] * 3 + [1])).to_anatomy_collection())
    body.attach_scan(from_nifti(tmp_path / "ct.nii.gz"), source_path=tmp_path / "ct.nii.gz")
    target = body.export_patient_twin(tmp_path / "bundle", vessel_names=["aorta"], ct_exterior=True)

    stage = Usd.Stage.Open(str(target.parent / "patient_anatomy.usdc"))
    skin = UsdGeom.Mesh(stage.GetPrimAtPath("/HumanBody/Exterior/CT"))
    faces = np.asarray(skin.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    edges = np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    count = len(skin.GetPointsAttr().Get())
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    assert connected_components(graph, directed=False)[0] == 1


@pytest.mark.parametrize("shift_mm", [0.0, 10.0])
def test_navigation_mask_follows_attached_registration(tmp_path, shift_mm):
    """A custom attach_scan registration moves the navigation path with the exported vessel."""
    from patient_digital_twin.scan_volume import from_array

    labels = np.zeros((30, 20, 25), np.uint8)  # ZYX on a 1 mm grid.
    labels[3:27, 8:13, 9:14] = 1  # Aorta occupies X indices 9-13.
    body = HumanBody(SegmentationImporter(labels, {1: "aorta"}).to_anatomy_collection())
    registration = body.anatomy.body_to_imaging.copy()
    registration[0, 3] += shift_mm / 1000
    ct = from_array(np.where(labels, 300.0, 40.0).transpose(2, 1, 0), np.eye(4), world_unit="mm")
    body.attach_scan(ct, body_to_imaging=registration)
    manifest = body.export_patient_twin(tmp_path / "bundle", vessel_names=["aorta"])
    x = np.load(manifest.parent / "centerline_points.npy")[:, 0]
    np.testing.assert_allclose(x, 11.0 + shift_mm, atol=1e-6)
    mask = np.load(manifest.parent / "vessel_mask.npy")  # Native IJK order from from_array.
    assert mask[int(9 + shift_mm):int(14 + shift_mm)].any() and mask.sum() == labels.sum()
