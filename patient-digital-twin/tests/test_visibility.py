# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct visibility changes preserve stored mesh geometry."""

import numpy as np
import pytest
from patient_digital_twin import (
    HumanBody,
    System,
)
from patient_digital_twin.importers._segmentation import anatomy_from_labels


@pytest.fixture
def body():
    result = HumanBody(
        anatomy_from_labels(["liver", "pancreas", "femur_left", "heart", "spleen"])
    )
    for structure in list(result.anatomy.structures.values())[:-1]:
        structure.vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        structure.faces = np.array([[0, 1, 2]])
    return result


def test_master_switch_preserves_mesh_and_metadata(body):
    liver = body.anatomy.structures["liver"]
    vertices, faces = liver.vertices, liver.faces
    body.anatomy.set_system_enabled("digestive", False)
    body.anatomy.set_structure_enabled("liver", True)
    body.anatomy.set_enabled(False)
    for structure in body.anatomy.structures.values():
        assert structure.is_empty
        assert structure.vertices is structure.faces is None
        assert structure.body_vertices is structure.world_vertices is None
    assert body.anatomy.select(include_empty=False) == []
    assert len(body.anatomy.select()) == 5
    assert liver.mesh.vertices is vertices
    body.anatomy.set_enabled(True)
    assert liver.vertices is vertices and liver.faces is faces
    assert not body.anatomy.structures["pancreas"].is_empty  # Last call wins.
    assert body.anatomy.structures[
        "spleen"
    ].is_empty  # Enabling never invents geometry.


def test_system_view_multisystem_rules_and_structure_override(body):
    digestive = body.anatomy.system(System.DIGESTIVE)
    assert [s.name for s in digestive.structures] == ["liver", "pancreas"]
    assert not digestive.is_empty
    body.anatomy.set_system_enabled("endocrine", False)
    assert body.anatomy.structures["pancreas"].is_empty
    assert not body.anatomy.structures["liver"].is_empty
    digestive.set_enabled(False)
    assert digestive.is_empty
    body.anatomy.set_structure_enabled("pancreas", True)
    assert not digestive.is_empty
    assert not body.anatomy.structures["pancreas"].is_empty
    assert not body.anatomy.system("skeletal").is_empty


def test_unknown_setter_targets_and_nonboolean_rejected(body):
    with pytest.raises(KeyError):
        body.anatomy.set_structure_enabled("not_an_organ", False)
    with pytest.raises(ValueError):
        body.anatomy.set_system_enabled("not_a_system", False)
    with pytest.raises(ValueError, match="boolean"):
        body.anatomy.set_enabled("false")


