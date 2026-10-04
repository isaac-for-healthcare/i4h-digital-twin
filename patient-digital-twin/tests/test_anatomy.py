# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog, name normalization, structures, and visibility controls."""

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
from patient_digital_twin.anatomy import CATALOG, canonical_name, is_vessel

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
