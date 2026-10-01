# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from patient_digital_twin import HumanBody, Kind, System
from patient_digital_twin.importers._labels import anatomy_from_labels


def test_human_body_owns_anatomy_without_an_external_body():
    from patient_digital_twin import AnatomyCollection

    anatomy = AnatomyCollection()
    body = HumanBody(anatomy)
    assert body.anatomy is anatomy
    assert body.imaging is None
    assert not hasattr(body, "source_path")
    assert not hasattr(body, "body_to_imaging")
    with pytest.raises(TypeError):
        HumanBody(source_path="scan.nii")
    with pytest.raises(TypeError):
        HumanBody(body_to_imaging=np.eye(4))
    with pytest.raises(TypeError):
        HumanBody(landmarks={})


def test_human_body_and_catalog():
    body = HumanBody(
        anatomy_from_labels({1: "liver", 2: "kidney_left", 3: "femur_left"})
    )
    assert isinstance(body, HumanBody)
    assert [s.name for s in body.anatomy.select(kind=Kind.BONE)] == ["femur_left"]
    assert [s.name for s in body.anatomy.select(system=System.URINARY)] == [
        "kidney_left"
    ]
    assert body.anatomy.structures["liver"].vertices is None


def test_strict_label_import_is_atomic():
    body = HumanBody(anatomy_from_labels(["liver"]))
    with pytest.raises(ValueError, match="Unmapped"):
        HumanBody(
            anatomy_from_labels({2: "heart", 3: "unmapped"}, anatomy=body.anatomy)
        )
    assert list(body.anatomy.structures) == ["liver"]


def test_optional_imaging_transform_recovers_scan_coordinates_without_aliasing():
    from patient_digital_twin import AnatomicalStructure
    from patient_digital_twin.geometry import transform_points

    placement = np.eye(4)
    placement[:3, 3] = [0.2, 0.1, -0.3]
    structure = AnatomicalStructure(
        "liver",
        Kind.ORGAN,
        vertices=np.array([[0.0, 0, 0], [0.1, 0.2, 0.3]]),
        local_to_body=placement,
    )
    body = HumanBody({"liver": structure})
    assert body.imaging is None
    with pytest.raises(ValueError, match="attach_imaging"):
        body.imaging_vertices("liver")
    imaging = np.array([[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1.0]])
    body.attach_imaging(
        np.zeros((2, 3, 4)), voxel_to_imaging=np.eye(4), body_to_imaging=imaging
    )
    expected = transform_points(structure.body_vertices, imaging)
    imaging[:3, 3] = 99
    copy = body.imaging.body_to_imaging.copy()
    copy[:3, 3] = 100
    structure.local_to_world[:3, 3] = [-1, -2, -3]
    np.testing.assert_allclose(body.imaging_vertices("liver"), expected)
    body.anatomy.set_enabled(False)
    assert body.imaging_vertices("liver") is None
    with pytest.raises(ValueError, match="rigid"):
        body.attach_imaging(
            np.zeros((2, 3, 4)),
            voxel_to_imaging=np.eye(4),
            body_to_imaging=np.diag([2, 1, 1, 1]),
        )
    np.testing.assert_allclose(body.imaging.body_to_imaging[:3, 3], [2, 3, 4])


def test_attach_imaging_owns_volume_and_metadata_without_loading_provenance():
    body = HumanBody()
    body.anatomy.body_to_imaging = np.eye(4)
    body.anatomy.source_path = "segmentation.nii.gz"
    assert body.imaging is None
    volume = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    affine = np.diag([0.001, 0.002, 0.003, 1])
    result = body.attach_imaging(
        volume, voxel_to_imaging=affine, source_path="not-a-file.nii"
    )
    assert result is body.imaging
    assert result.source_path == "not-a-file.nii"
    volume[:] = -1
    affine[0, 0] = 99
    np.testing.assert_array_equal(result.volume.ravel(), np.arange(24))
    assert result.voxel_to_imaging[0, 0] == 0.001
    with pytest.raises(ValueError):
        result.volume[0, 0, 0] = 7
    with pytest.raises(ValueError):
        result.body_to_imaging[0, 0] = 2
    assert body.anatomy.source_path == "segmentation.nii.gz"


@pytest.mark.parametrize(
    "volume",
    [
        np.zeros((2, 3)),
        np.zeros((0, 2, 3)),
        np.full((2, 2, 2), np.nan),
        np.ones((2, 2, 2), dtype=complex),
        np.ones((2, 2, 2), dtype=bool),
        np.full((2, 2, 2), "bad"),
    ],
)
def test_invalid_imaging_preserves_existing_attachment(volume):
    body = HumanBody()
    valid = body.attach_imaging(
        np.zeros((2, 3, 4)), voxel_to_imaging=np.eye(4), body_to_imaging=np.eye(4)
    )
    with pytest.raises(ValueError):
        body.attach_imaging(
            volume, voxel_to_imaging=np.eye(4), body_to_imaging=np.eye(4)
        )
    assert body.imaging is valid


def test_imaging_requires_explicit_spatial_metadata_and_numpy_volume():
    body = HumanBody()
    with pytest.raises(ValueError, match="registration"):
        body.attach_imaging(np.zeros((2, 3, 4)), voxel_to_imaging=np.eye(4))
    with pytest.raises(TypeError, match="NumPy"):
        body.attach_imaging(
            "scan.nii", voxel_to_imaging=np.eye(4), body_to_imaging=np.eye(4)
        )
    for affine in (np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan)):
        with pytest.raises(ValueError, match="affine"):
            body.attach_imaging(
                np.zeros((2, 3, 4)), voxel_to_imaging=affine, body_to_imaging=np.eye(4)
            )
    assert body.imaging is None
