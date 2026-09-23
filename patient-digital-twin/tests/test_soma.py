# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from patient_digital_twin import HumanBody
import numpy as np
import pytest
from patient_digital_twin.importers._labels import anatomy_from_labels

torch = pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")

from patient_digital_twin.geometry import rigid_transform


class ArticulatedLayer(torch.nn.Module):
    """Small deterministic SOMA-protocol fixture, not a real SOMA model."""

    public_joint_names = (
        "Root",
        "Hips",
        "Spine",
        "Spine1",
        "Spine2",
        "LeftArm",
        "RightArm",
    )
    scale_param_names = ()
    num_scale_params = 0
    num_shape_components = 1
    output_unit = "meters"

    def __init__(self):
        super().__init__()
        self.register_buffer("dummy", torch.zeros(1))
        skin = trimesh.creation.icosphere(subdivisions=2, radius=1)
        self.register_buffer("faces", torch.as_tensor(skin.faces.copy()))
        self.skin = torch.as_tensor(skin.vertices.copy(), dtype=torch.float32)

    def forward(self, poses, identity_coeffs, transl, **kwargs):
        from scipy.spatial.transform import Rotation

        rotation = torch.as_tensor(
            Rotation.from_rotvec(poses[0, 0].numpy()).as_matrix(), dtype=torch.float32
        )
        matrix = torch.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotation, transl[0]
        transforms = matrix.repeat(1, len(self.public_joint_names), 1, 1)
        # Deliberately different Root proves that indexing includes Root.
        transforms[:, 0] = torch.eye(4)
        transforms[:, 1] = torch.eye(4)
        return {
            "vertices": (self.skin @ rotation.T + transl)[None],
            "transforms": transforms,
        }

    def public_skinning_weights(self):
        weights = torch.zeros(len(self.skin), len(self.public_joint_names))
        weights[:, 1] = 0.3
        weights[:, 2] = 0.7
        return weights


@pytest.fixture
def bound_body():
    body = HumanBody(anatomy_from_labels(["liver", "kidney_left"]))
    for index, structure in enumerate(body.anatomy.structures.values()):
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.08)
        structure.vertices = mesh.vertices.copy()
        structure.faces = mesh.faces.copy()
        structure.local_to_body[:3, 3] = [0.15 * index, 0.1, 0]
    body.AttachExternalBody(
        ArticulatedLayer(),
        body_to_soma=np.eye(4),
        anchors={"liver": "Spine", "kidney_left": "Spine"},
    )
    return body


@pytest.mark.parametrize(
    "rotation", [[0, 0, 0], [0.8, 0.2, 0], [0, 0, 1.4], [-0.5, 0.4, 0.2]]
)
def test_rigid_anchors_and_containment(bound_body, rotation):
    original = {n: s.body_vertices.copy() for n, s in bound_body.anatomy.structures.items()}
    poses = np.zeros((1, 6, 3))
    poses[0, 0] = rotation
    bound_body.soma.pose(poses, transl=[[0.3, -0.2, 0.8]])
    for name, structure in bound_body.anatomy.structures.items():
        rigid_transform(structure.local_to_world)
        np.testing.assert_allclose(structure.body_vertices, original[name])
        np.testing.assert_allclose(
            structure.local_to_world,
            bound_body.soma.soma_body.transforms["Spine"] @ structure.local_to_anchor,
        )
    assert all(item["passed"] for item in bound_body.soma.check_containment().values())


def test_direct_forward_updates_anchors(bound_body):
    params = bound_body.soma.soma_parameters
    params["transl"] = torch.tensor([[1.0, 2.0, 3.0]])
    bound_body.soma.soma_layer(**params)
    structure = bound_body.anatomy.structures["liver"]
    np.testing.assert_allclose(
        structure.world_vertices, structure.body_vertices + [1, 2, 3], atol=1e-6
    )


@pytest.mark.parametrize("disable_before_attach", [False, True])
def test_disabled_meshes_restore_at_current_soma_pose(
    bound_body, disable_before_attach
):
    body = bound_body
    structure = body.anatomy.structures["liver"]
    source = structure.vertices.copy()
    if disable_before_attach:
        body.anatomy.set_enabled(False)
        body.AttachExternalBody(body.soma.soma_layer, **body.soma.soma_configuration)
    poses = np.zeros((1, 6, 3))
    poses[0, 0] = [0.2, 0.1, 0.3]
    body.anatomy.set_enabled(True)
    body.soma.pose(poses)
    expected = structure.world_vertices.copy()
    skin = body.soma.soma_body.vertices.copy()
    body.soma.pose()
    body.anatomy.set_enabled(False)
    parameters = body.soma.soma_parameters
    parameters["poses"] = torch.as_tensor(poses, dtype=torch.float32)
    body.soma.soma_layer(**parameters)  # Direct calls must also update disabled storage.
    assert body.soma.check_containment() == {}
    assert structure.vertices is structure.faces is None
    np.testing.assert_allclose(body.soma.soma_body.vertices, skin)
    body.anatomy.set_enabled(True)
    np.testing.assert_allclose(structure.world_vertices, expected)
    np.testing.assert_array_equal(structure.vertices, source)
    assert all(item["passed"] for item in body.soma.check_containment().values())


def test_containment_reports_failure_and_open_skin(bound_body):
    structure = bound_body.anatomy.structures["liver"]
    structure.local_to_world[:3, 3] = [5, 0, 0]
    assert not bound_body.soma.check_containment()["liver"]["passed"]
    bound_body.soma.soma_body.faces[0] = bound_body.soma.soma_body.faces[1]
    with pytest.raises(ValueError, match="watertight"):
        bound_body.soma.check_containment()


def test_alignment_cannot_be_guessed(bound_body):
    with pytest.raises(ValueError, match="at least 3"):
        HumanBody(anatomy_from_labels(["liver"])).AttachExternalBody(ArticulatedLayer())


def test_bone_registration_and_configuration_roundtrip(bound_body):
    import json

    HumanBody(anatomy_from_labels({3: "femur_left"}, anatomy=bound_body.anatomy))
    bone = bound_body.anatomy.structures["femur_left"]
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.08)
    bone.vertices, bone.faces = mesh.vertices.copy(), mesh.faces.copy()
    bone.local_to_body[:3, 3] = [0.98, 0, 0]
    bound_body.AttachExternalBody(
        bound_body.soma.soma_layer,
        body_to_soma=np.eye(4),
        anchors={name: "Spine" for name in bound_body.anatomy.structures},
    )
    original = bone.vertices.copy()
    pose = np.zeros((1, 6, 3))
    pose[0, 0, 2] = 0.7
    report = bound_body.soma.fit_bone_anchors([pose], max_translation_m=0.1)
    assert report["femur_left"]["final_protrusion_m"] == 0
    np.testing.assert_array_equal(bone.vertices, original)
    rigid_transform(bone.local_to_anchor)
    assert bound_body.soma.check_containment()["femur_left"]["passed"]
    saved = json.loads(json.dumps(bound_body.soma.soma_configuration))
    expected = bone.world_vertices.copy()
    bound_body.AttachExternalBody(bound_body.soma.soma_layer, **saved)
    np.testing.assert_allclose(bone.world_vertices, expected)


def test_torso_does_not_follow_nearby_arm(bound_body):
    before = bound_body.anatomy.structures["liver"].world_vertices.copy()
    layer = bound_body.soma.soma_layer
    output = layer.forward(**bound_body.soma.soma_parameters)
    output["transforms"][:, layer.public_joint_names.index("LeftArm"), :3, 3] += 2
    bound_body.soma.update_from_soma(output)
    np.testing.assert_allclose(bound_body.anatomy.structures["liver"].world_vertices, before)
