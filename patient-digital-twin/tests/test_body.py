# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog, structures, visibility, and attached imaging."""

import numpy as np
import pytest
from patient_digital_twin import (
    AnatomicalStructure,
    AnatomyCollection,
    HumanBody,
    Kind,
    MeshGeometry,
    System,
)
from patient_digital_twin.body import CATALOG, canonical_name, is_vessel
from patient_digital_twin.geometry import transform_points
from patient_digital_twin.scan_volume import from_array

TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@pytest.fixture
def body():
    result = HumanBody(AnatomyCollection.from_names(["liver", "pancreas", "femur_left", "heart", "spleen"]))
    for structure in list(result.anatomy.structures.values())[:-1]:
        structure.vertices, structure.faces = TRIANGLE.copy(), np.array([[0, 1, 2]])
    return result


@pytest.mark.parametrize("enum", [Kind, System])
def test_enum_names_roundtrip_and_reject_unknown(enum):
    assert len({item.value for item in enum}) == len(enum)
    assert all(enum(item.value) is item and isinstance(item, str) for item in enum)
    with pytest.raises(ValueError):
        enum("not_a_supported_name")


def test_catalog_kinds_systems_and_name_normalization():
    assert len(CATALOG) == 122
    assert CATALOG["pancreas"] == (Kind.ORGAN, {System.DIGESTIVE, System.ENDOCRINE})
    assert CATALOG["rib_right_12"][0] is Kind.BONE and "vertebrae_L6" in CATALOG
    assert is_vessel("aorta") and is_vessel("portal_vein_and_splenic_vein") and not is_vessel("liver")
    for raw, name in [("Left Kidney", "kidney_left"), ("rib 3 R", "rib_right_3"), ("bladder", "urinary_bladder"),
                      ("vertebrae_t12", "vertebrae_T12"), ("Adrenal-gland r", "adrenal_gland_right")]:
        assert canonical_name(raw) == name


def test_from_names_shares_duplicates_and_rejects_unknown_when_strict():
    anatomy = AnatomyCollection.from_names(["liver", "liver", "heart"])
    assert list(anatomy.structures) == ["liver", "heart"] and anatomy.structures["liver"].kind is Kind.ORGAN
    assert AnatomyCollection.from_names(["custom"], strict=False).structures["custom"].kind is Kind.UNKNOWN
    with pytest.raises(ValueError, match="Unmapped"):
        AnatomyCollection.from_names(["custom"])


def test_mesh_storage_and_default_transforms_are_independent():
    mesh, other = MeshGeometry(), MeshGeometry()
    mesh.vertices = TRIANGLE
    assert other.vertices is None and mesh.vertices is TRIANGLE
    first, second = AnatomicalStructure("first", Kind.UNKNOWN), AnatomicalStructure("second", Kind.UNKNOWN)
    first.local_to_body[0, 3] = 1
    assert first.is_empty and first.world_vertices is None
    assert second.local_to_body[0, 3] == 0 and first.mesh is not second.mesh


def test_structure_coordinate_properties_and_reversible_visibility():
    imaging, world = np.eye(4), np.eye(4)
    imaging[:3, 3], world[:3, 3] = [1, 2, 3], [4, 5, 6]
    structure = AnatomicalStructure("liver", Kind.ORGAN, TRIANGLE, np.array([[0, 1, 2]]), imaging, world)
    np.testing.assert_array_equal(structure.body_vertices, TRIANGLE + [1, 2, 3])
    np.testing.assert_array_equal(structure.world_vertices, TRIANGLE + [4, 5, 6])
    structure.enabled = False
    assert structure.vertices is structure.faces is None
    assert structure.body_vertices is structure.world_vertices is None
    assert structure.mesh.vertices is TRIANGLE
    structure.enabled = True
    structure.centerline = object()
    structure.vertices = TRIANGLE.copy()
    assert structure.centerline is None


def test_master_switch_preserves_mesh_and_last_call_wins(body):
    liver = body.anatomy.structures["liver"]
    vertices, faces = liver.vertices, liver.faces
    body.anatomy.set_system_enabled("digestive", False)
    body.anatomy.set_structure_enabled("liver", True)
    body.anatomy.set_enabled(False)
    assert all(s.is_empty for s in body.anatomy.structures.values())
    assert body.anatomy.select(include_empty=False) == [] and len(body.anatomy.select()) == 5
    body.anatomy.set_enabled(True)
    assert liver.vertices is vertices and liver.faces is faces
    assert not body.anatomy.structures["pancreas"].is_empty
    assert body.anatomy.structures["spleen"].is_empty  # Enabling never invents geometry.


def test_system_selection_and_multisystem_structures(body):
    anatomy = body.anatomy
    assert [s.name for s in anatomy.select(system=System.DIGESTIVE)] == ["liver", "pancreas"]
    assert [s.name for s in anatomy.select(kind=Kind.BONE)] == ["femur_left"]
    anatomy.set_system_enabled("endocrine", False)
    assert anatomy.structures["pancreas"].is_empty and not anatomy.structures["liver"].is_empty
    with pytest.raises(KeyError):
        anatomy.set_structure_enabled("not_an_organ", False)
    with pytest.raises(ValueError):
        anatomy.set_system_enabled("not_a_system", False)
    with pytest.raises(ValueError, match="boolean"):
        anatomy.set_enabled("false")


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
