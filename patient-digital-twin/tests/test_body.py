# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HumanBody ownership of anatomy and attached imaging."""

import numpy as np
import pytest
from patient_digital_twin import AnatomicalStructure, AnatomyCollection, HumanBody, Kind
from patient_digital_twin.geometry import transform_points
from patient_digital_twin.scan_volume import from_array

TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_human_body_owns_anatomy():
    anatomy = AnatomyCollection()
    body = HumanBody(anatomy)
    assert body.anatomy is anatomy and body.imaging is None
    assert HumanBody({"liver": AnatomicalStructure("liver", Kind.ORGAN)}).anatomy.structures["liver"]
    with pytest.raises(TypeError):
        HumanBody(source_path="scan.nii")


def test_attach_imaging_copies_volume_and_validates_registration():
    structure = AnatomicalStructure("liver", Kind.ORGAN, TRIANGLE, np.array([[0, 1, 2]]))
    body = HumanBody({"liver": structure})
    with pytest.raises(ValueError, match="registration"):
        body.attach_imaging(np.zeros((2, 3, 4)), voxel_to_imaging=np.eye(4))
    volume = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    affine, imaging = np.diag([0.001, 0.002, 0.003, 1]), np.eye(4)
    imaging[:3, 3] = [2, 3, 4]
    result = body.attach_imaging(volume, voxel_to_imaging=affine, body_to_imaging=imaging, source_path="x.nii")
    assert result is body.imaging and result.source_path == "x.nii"
    volume[:], affine[0, 0], imaging[:3, 3] = -1, 99, 99
    np.testing.assert_array_equal(result.volume.ravel(), np.arange(24))
    np.testing.assert_allclose(result.voxel_to_imaging, np.diag([0.001, 0.002, 0.003, 1]))
    np.testing.assert_allclose(transform_points(structure.body_vertices, result.body_to_imaging), TRIANGLE + [2, 3, 4])
    for array in (result.volume, result.body_to_imaging):
        with pytest.raises(ValueError):
            array[0, 0] = 7
    with pytest.raises(ValueError, match="rigid"):
        body.attach_imaging(np.zeros((2, 3, 4)), voxel_to_imaging=np.eye(4), body_to_imaging=np.diag([2, 1, 1, 1]))
    assert body.imaging is result


@pytest.mark.parametrize(
    "volume, affine",
    [
        (np.zeros((2, 3)), np.eye(4)),
        (np.zeros((0, 2, 3)), np.eye(4)),
        (np.full((2, 2, 2), np.nan), np.eye(4)),
        (np.ones((2, 2, 2), dtype=complex), np.eye(4)),
        (np.ones((2, 2, 2), dtype=bool), np.eye(4)),
        ("scan.nii", np.eye(4)),
        (np.zeros((2, 3, 4)), np.zeros((4, 4))),
        (np.zeros((2, 3, 4)), np.eye(3)),
        (np.zeros((2, 3, 4)), np.full((4, 4), np.nan)),
    ],
)
def test_invalid_imaging_preserves_existing_attachment(volume, affine):
    body = HumanBody()
    valid = body.attach_imaging(np.zeros((2, 3, 4)), voxel_to_imaging=np.eye(4), body_to_imaging=np.eye(4))
    with pytest.raises(ValueError):
        body.attach_imaging(volume, voxel_to_imaging=affine, body_to_imaging=np.eye(4))
    assert body.imaging is valid


def test_attach_scan_shares_native_buffer_and_keeps_axes():
    scan = from_array(np.arange(24).reshape(2, 3, 4), np.diag([2, 3, 4, 1]),
                      array_axes="ijk", world_frame="LPS", world_unit="mm")
    imaging = HumanBody().attach_scan(scan, body_to_imaging=np.eye(4))
    assert imaging.scan is scan and np.shares_memory(imaging.volume, scan.values)
    assert imaging.volume.shape == (4, 3, 2) and not imaging.volume.flags.writeable
    np.testing.assert_allclose(imaging.voxel_to_imaging, scan.ijk_to_ras_m)
    with pytest.raises(TypeError, match="ScanVolume"):
        HumanBody().attach_scan(np.zeros((2, 2, 2)), body_to_imaging=np.eye(4))
