# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mesh policy tests require no SOMA, viewer, or imaging mesher."""

import numpy as np
import pytest
from patient_digital_twin import (
    AnatomyConfiguration,
    HumanBody,
    System,
)
from patient_digital_twin.importers._labels import anatomy_from_labels


@pytest.fixture
def body():
    result = HumanBody(anatomy_from_labels(["liver", "pancreas", "femur_left", "heart", "spleen"]))
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
    assert body.anatomy.structures["pancreas"].is_empty  # Earlier system policy survives.
    assert body.anatomy.structures["spleen"].is_empty  # Enabling never invents geometry.


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


def test_yaml_file_and_mapping_roundtrip(body, tmp_path):
    import yaml

    path = tmp_path / "anatomy.yaml"
    path.write_text(
        "anatomy:\n  enabled: true\n  systems:\n    digestive: false\n"
        "  structures:\n    liver: true\n",
        encoding="utf-8",
    )
    body.anatomy.configure(path)
    assert not body.anatomy.structures["liver"].is_empty
    assert body.anatomy.structures["pancreas"].is_empty
    policy = body.anatomy.configuration
    assert AnatomyConfiguration.load(policy) is policy
    restored = AnatomyConfiguration.load(
        yaml.safe_load(yaml.safe_dump(policy.to_dict()))
    )
    assert restored == policy
    with pytest.raises(TypeError):
        policy.systems[System.SKELETAL] = False
    body.anatomy.configure({})  # Replacement, not a merge.
    assert not body.anatomy.structures["pancreas"].is_empty


@pytest.mark.parametrize(
    "config",
    [
        {"anatomy": {"enabled": "false"}},
        {"anatomy": {"enabled": 0}},
        {"anatomy": {"systems": {"digestive": "false"}}},
        {"anatomy": {"systems": {"typo": False}}},
        {"anatomy": {"structures": {"livre": False}}},
        {"anatomy": {"structures": {"liver": None}}},
        {"anatomy": {"systems": []}},
        {"anatomy": {"unknown": True}},
        {"anatomy": None},
        {"unexpected": {}},
        [],
    ],
)
def test_invalid_configuration_is_atomic(body, config):
    policy = body.anatomy.configuration
    vertices = body.anatomy.structures["liver"].vertices
    with pytest.raises((TypeError, ValueError)):
        body.anatomy.configure(config)
    assert body.anatomy.configuration is policy
    assert body.anatomy.structures["liver"].vertices is vertices


@pytest.mark.parametrize(
    "text",
    [
        "anatomy:\n  enabled: true\n  enabled: false\n",
        "anatomy: {systems: {digestive: false, digestive: true}}",
    ],
)
def test_duplicate_yaml_keys_rejected(body, tmp_path, text):
    path = tmp_path / "duplicate.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        body.anatomy.configure(path)


def test_configuration_applies_to_later_imports_and_meshes():
    body = HumanBody(configuration={"anatomy": {"enabled": False}})
    HumanBody(anatomy_from_labels(["liver"], anatomy=body.anatomy))
    liver = body.anatomy.structures["liver"]
    liver.vertices = np.ones((3, 3))
    liver.faces = np.array([[0, 1, 2]])
    assert liver.is_empty
    body.anatomy.set_enabled(True)
    assert not liver.is_empty


def test_unknown_setter_targets_and_nonboolean_rejected(body):
    with pytest.raises(ValueError, match="Unknown"):
        body.anatomy.set_structure_enabled("not_an_organ", False)
    with pytest.raises(ValueError):
        body.anatomy.set_system_enabled("not_a_system", False)
    with pytest.raises(ValueError, match="boolean"):
        body.anatomy.set_enabled("false")


def test_yaml_loader_rejects_object_tags_and_nonstring_keys(body, tmp_path):
    import yaml

    path = tmp_path / "invalid.yaml"
    path.write_text("anatomy: !!python/object:builtins.object {}", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError):
        body.anatomy.configure(path)
    path.write_text("anatomy: {structures: {1: false}}", encoding="utf-8")
    with pytest.raises(TypeError, match="keys must be strings"):
        body.anatomy.configure(path)
    assert not body.anatomy.structures["liver"].is_empty


def test_empty_yaml_restores_defaults(body, tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    body.anatomy.set_enabled(False)
    body.anatomy.configure(path)
    assert body.anatomy.configuration.enabled
    assert not body.anatomy.structures["liver"].is_empty
