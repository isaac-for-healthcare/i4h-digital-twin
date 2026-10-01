# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct contracts for metadata, enums, and retained mesh storage."""

import numpy as np
import pytest
from patient_digital_twin import (
    AnatomicalStructure,
    Kind,
    MeshGeometry,
    System,
)


@pytest.mark.parametrize("enum", [Kind, System])
def test_enum_names_roundtrip_and_reject_unknown(enum):
    assert len({item.value for item in enum}) == len(enum)
    for item in enum:
        assert enum(item.value) is item
        assert isinstance(item, str)
    with pytest.raises(ValueError):
        enum("not_a_supported_name")


def test_mesh_geometry_defaults_and_retained_arrays():
    mesh, other = MeshGeometry(), MeshGeometry()
    assert mesh.vertices is mesh.faces is None
    vertices, faces = np.zeros((3, 3)), np.array([[0, 1, 2]])
    mesh.vertices, mesh.faces = vertices, faces
    assert mesh.vertices is vertices and mesh.faces is faces
    assert other.vertices is None


def test_structure_default_storage_is_independent():
    first = AnatomicalStructure("first", Kind.UNKNOWN)
    second = AnatomicalStructure("second", Kind.UNKNOWN)
    first.local_to_body[0, 3] = 1
    assert first.is_empty and first.world_vertices is None
    assert second.local_to_body[0, 3] == 0
    assert first.mesh is not second.mesh


def test_structure_coordinate_properties_and_reversible_visibility():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    imaging, world = np.eye(4), np.eye(4)
    imaging[:3, 3], world[:3, 3] = [1, 2, 3], [4, 5, 6]
    structure = AnatomicalStructure(
        "liver",
        Kind.ORGAN,
        vertices=vertices,
        faces=np.array([[0, 1, 2]]),
        local_to_body=imaging,
        local_to_world=world,
    )
    np.testing.assert_array_equal(structure.body_vertices, vertices + [1, 2, 3])
    np.testing.assert_array_equal(structure.world_vertices, vertices + [4, 5, 6])
    structure.enabled = False
    assert structure.vertices is structure.faces is None
    assert structure.body_vertices is structure.world_vertices is None
    assert structure.mesh.vertices is vertices
    structure.enabled = True
    structure.vertices = vertices.copy()
    np.testing.assert_array_equal(structure.world_vertices, vertices + [4, 5, 6])
