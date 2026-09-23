# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct SOMA owner/lifecycle tests with a deterministic CPU model fixture."""

from patient_digital_twin import HumanBody
import json

import numpy as np
import pytest
from patient_digital_twin import AnatomyCollection, SomaRepresentation

torch = pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")
from test_soma import ArticulatedLayer
from test_soma import bound_body as bound_body  # noqa: PLC0414 - pytest fixture export


@pytest.mark.parametrize(
    "method,args",
    [
        ("pose", ()),
        ("update_from_soma", ({},)),
        ("fit_soma_shape", ()),
        ("fit_bone_anchors", ()),
        ("check_containment", ()),
    ],
)
def test_unattached_representation_rejects_model_operations(method, args):
    anatomy = AnatomyCollection()
    soma = SomaRepresentation(anatomy)
    assert soma.structures is anatomy.structures
    assert soma.soma_layer is soma.soma_body is None
    assert soma.soma_parameters == {}
    with pytest.raises(RuntimeError, match="attach_soma"):
        getattr(soma, method)(*args)
    with pytest.raises(RuntimeError, match="attach_soma"):
        _ = soma.soma_configuration


def test_parameters_are_copies_and_pose_does_not_change_scan_baseline(bound_body):
    soma = bound_body.soma
    assert isinstance(soma, SomaRepresentation)
    scan = soma.soma_body.vertices.copy()
    params = soma.soma_parameters
    params["poses"].add_(1)
    assert not torch.equal(params["poses"], soma.soma_parameters["poses"])
    soma.pose(transl=[[0.1, 0.2, 0.3]])
    assert not np.allclose(scan, soma.soma_body.vertices)
    soma.pose()
    np.testing.assert_allclose(scan, soma.soma_body.vertices)
    saved = json.loads(json.dumps(soma.soma_configuration))
    soma.attach_soma(soma.soma_layer, **saved)
    np.testing.assert_allclose(scan, soma.soma_body.vertices)


def test_failed_reattachment_preserves_old_bindings_and_hook(bound_body):
    soma = bound_body.soma
    original_layer, original_pose = soma.soma_layer, soma.soma_body
    offset = soma.structures["liver"].local_to_anchor.copy()
    with pytest.raises(ValueError, match="Unknown SOMA anchor"):
        soma.attach_soma(
            ArticulatedLayer(),
            body_to_soma=np.eye(4),
            anchors={"liver": "missing"},
        )
    assert soma.soma_layer is original_layer and soma.soma_body is original_pose
    np.testing.assert_array_equal(offset, soma.structures["liver"].local_to_anchor)
    assert len(original_layer._forward_hooks) == 1
    replacement = ArticulatedLayer()
    soma.attach_soma(replacement, **soma.soma_configuration)
    assert len(original_layer._forward_hooks) == 0
    assert len(replacement._forward_hooks) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"components": 0},
        {"iterations": 0},
        {"margin_m": 0},
    ],
)
def test_shape_fit_rejects_invalid_options(bound_body, options):
    with pytest.raises(ValueError, match="Positive"):
        bound_body.soma.fit_soma_shape(**options)


def test_shape_optimizer_changes_skin_not_source_geometry(bound_body):
    class ShapeResponsiveLayer(ArticulatedLayer):
        """A single identity coefficient expands the synthetic skin radially."""

        def forward(self, poses, identity_coeffs, transl, **kwargs):
            result = super().forward(poses, identity_coeffs, transl, **kwargs)
            center = transl[:, None, :]
            factor = 1 + 0.2 * identity_coeffs[0, 0]
            result["vertices"] = (result["vertices"] - center) * factor + center
            return result

    body = bound_body
    body.anatomy.structures["liver"].vertices += [0.95, 0, 0]
    originals = {name: s.vertices.copy() for name, s in body.anatomy.structures.items()}
    body.AttachExternalBody(ShapeResponsiveLayer(), **body.soma.soma_configuration)
    assert not body.soma.check_containment()["liver"]["passed"]
    report = body.soma.fit_soma_shape(components=1, iterations=4)
    assert report is body.soma.shape_fit_report
    assert report["identity_coeffs"][0] > 0
    assert report["history"][0]["active_points"] > 0
    assert all(result["passed"] for result in body.soma.check_containment().values())
    for name, structure in body.anatomy.structures.items():
        np.testing.assert_array_equal(structure.vertices, originals[name])
    assert len(body.soma.soma_layer._forward_hooks) == 1


def test_bone_landmark_matching_discards_hints_and_poses_all_anatomy_rigidly(
    monkeypatch,
):
    from patient_digital_twin import AnatomicalStructure, HumanBody, Kind
    from patient_digital_twin.geometry import transform_points
    from patient_digital_twin.soma_body.geometry import SOMA_JOINT_NAMES

    class LandmarkLayer(ArticulatedLayer):
        public_joint_names = (
            "Root",
            "Hips",
            "Spine1",
            "Spine2",
            "LeftArm",
            "RightArm",
            "LeftLeg",
            "RightLeg",
        )

        def forward(self, poses, identity_coeffs, transl, **kwargs):
            result = super().forward(poses, identity_coeffs, transl, **kwargs)
            centers = {
                "LeftArm": [-0.3, 0.7, 0],
                "RightArm": [0.3, 0.7, 0],
                "LeftLeg": [-0.2, 0, 0],
                "RightLeg": [0.2, 0, 0],
            }
            for name, center in centers.items():
                result["transforms"][0, self.public_joint_names.index(name), :3, 3] += (
                    torch.tensor(center)
                )
            return result

    alignment = np.array(
        [[0.0, -1, 0, 0.2], [1, 0, 0, -0.1], [0, 0, 1, 0.3], [0, 0, 0, 1]]
    )
    layer = LandmarkLayer()
    measured = {
        name: transform_points(point, np.linalg.inv(alignment))
        for name, point in {
            "left_shoulder": [-0.3, 0.7, 0],
            "right_shoulder": [0.3, 0.7, 0],
            "left_hip": [-0.2, 0, 0],
            "right_hip": [0.2, 0, 0],
        }.items()
    }
    vertices = np.array([[0.0, 0, 0], [0.02, 0, 0], [0, 0.03, 0]])
    body = HumanBody(
        {
            "humerus_left": AnatomicalStructure(
                "humerus_left", Kind.BONE, vertices=vertices.copy()
            ),
            "liver": AnatomicalStructure("liver", Kind.ORGAN, vertices=vertices.copy()),
        },
    )
    baseline = {
        name: s.body_vertices.copy() for name, s in body.anatomy.structures.items()
    }
    body.AttachExternalBody(
        layer,
        landmarks=measured,
        poses=np.zeros((1, 7, 3)),
        global_scale=1,
        refine_limbs=False,
    )
    np.testing.assert_allclose(body.soma.body_to_soma, alignment, atol=1e-7)
    assert (
        body.soma.landmarks == {} and measured
    )  # Do not clear the caller's dictionary.
    assert body.anatomy.structures["humerus_left"].anchor_joint == "LeftArm"
    assert body.anatomy.structures["liver"].anchor_joint in ("Spine1", "Spine2")
    for name, point in measured.items():
        np.testing.assert_allclose(
            transform_points(point, body.soma.body_to_soma),
            body.soma.soma_body.joints[SOMA_JOINT_NAMES[name]],
            atol=1e-7,
        )
    for name, structure in body.anatomy.structures.items():
        np.testing.assert_allclose(
            structure.world_vertices,
            transform_points(baseline[name], alignment),
            atol=1e-7,
        )

    def no_matching(*args, **kwargs):
        raise AssertionError("Posing must not perform anatomical matching")

    monkeypatch.setattr(
        "patient_digital_twin.soma_body.representation.default_anchor", no_matching
    )
    pose = np.zeros((1, 7, 3))
    pose[0, 0] = [0.4, 0.2, -0.3]
    body.soma.pose(pose, transl=[[0.1, -0.2, 0.3]])
    for name, structure in body.anatomy.structures.items():
        np.testing.assert_allclose(structure.body_vertices, baseline[name])
        np.testing.assert_allclose(
            np.linalg.norm(
                structure.world_vertices[:, None] - structure.world_vertices[None, :],
                axis=-1,
            ),
            np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1),
            atol=1e-8,
        )
    config = json.loads(json.dumps(body.soma.soma_configuration))
    expected = {
        name: s.local_to_anchor.copy() for name, s in body.anatomy.structures.items()
    }
    body.AttachExternalBody(layer, **config)  # Explicit stored joints bypass matching.
    for name, structure in body.anatomy.structures.items():
        np.testing.assert_allclose(structure.local_to_anchor, expected[name])


def test_default_pose_lowers_arms_without_changing_imaging_registration(bound_body):
    soma = bound_body.soma
    source = soma.soma_configuration
    names = list(soma.soma_layer.public_joint_names)
    source["poses"][0][names.index("LeftArm") - 1] = [0.2, 0.3, 0.4]
    source["poses"][0][names.index("RightArm") - 1] = [-0.2, -0.3, -0.4]
    observed = []
    hook = soma.soma_layer.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs["poses"].clone()),
        with_kwargs=True,
    )
    try:
        soma.attach_soma(soma.soma_layer, **source)
        np.testing.assert_allclose(observed[-1], soma.scan_pose)
        np.testing.assert_allclose(soma.soma_parameters["poses"], source["poses"])
        soma.pose()
        np.testing.assert_allclose(observed[-1], soma.scan_pose)
        for joint, sign in (("LeftArm", -1), ("RightArm", 1)):
            np.testing.assert_allclose(
                soma.scan_pose[0, names.index(joint) - 1],
                [0, 0, sign * np.deg2rad(80)],
            )
    finally:
        hook.remove()


def test_human_attachment_extracts_bone_joints_and_aligns_without_landmark_inputs():
    from patient_digital_twin import AnatomicalStructure, HumanBody, Kind

    joint_centers = {
        "LeftArm": [-0.3, 0.7, 0],
        "RightArm": [0.3, 0.7, 0],
        "LeftLeg": [-0.2, 0, 0],
        "RightLeg": [0.2, 0, 0],
        "LeftForeArm": [-0.3, 1.0, 0],
        "RightForeArm": [0.3, 1.0, 0],
        "LeftShin": [-0.2, -0.5, 0],
        "RightShin": [0.2, -0.5, 0],
    }

    class JointLayer(ArticulatedLayer):
        public_joint_names = (
            "Root",
            "Hips",
            "Spine1",
            "Spine2",
            "LeftArm",
            "RightArm",
            "LeftLeg",
            "RightLeg",
            "LeftForeArm",
            "RightForeArm",
            "LeftShin",
            "RightShin",
        )

        def forward(self, **parameters):
            output = super().forward(**parameters)
            for name, center in joint_centers.items():
                output["transforms"][0, self.public_joint_names.index(name), :3, 3] = (
                    torch.tensor(center)
                )
            return output

    structures = {}
    for name, center, length in (
        ("humerus_left", [-0.3, 0.85, 0], 0.3),
        ("humerus_right", [0.3, 0.85, 0], 0.3),
        ("femur_left", [-0.2, -0.25, 0], 0.5),
        ("femur_right", [0.2, -0.25, 0], 0.5),
        ("vertebrae_L5", [0, 0.3, 0], 0.05),
    ):
        mesh = trimesh.creation.box(extents=[0.04, length, 0.04])
        placement = np.eye(4)
        placement[:3, 3] = np.array(center) + [1, 2, 3]
        structures[name] = AnatomicalStructure(
            name,
            Kind.BONE,
            mesh.vertices.copy(),
            mesh.faces.copy(),
            local_to_body=placement,
        )
    body = HumanBody(structures)
    body.anatomy.set_enabled(False)  # Visibility must not remove fitting evidence.
    original = {name: s.mesh.vertices.copy() for name, s in structures.items()}
    assert body.soma is None
    skin = body.AttachExternalBody(JointLayer(), global_scale=1, refine_limbs=False)
    assert skin is body.soma.soma_body
    assert body.soma.anatomy is body.anatomy
    expected = np.eye(4)
    expected[:3, 3] = [-1, -2, -3]
    np.testing.assert_allclose(body.soma.body_to_soma, expected, atol=1e-6)
    assert max(body.soma.alignment_errors_m.values()) < 1e-6
    assert body.soma.landmarks == {}
    for name, structure in structures.items():
        np.testing.assert_array_equal(structure.mesh.vertices, original[name])


def test_failed_first_attachment_does_not_publish_a_representation():
    from patient_digital_twin import HumanBody

    body = HumanBody()
    with pytest.raises(ValueError, match="non-collinear"):
        body.AttachExternalBody(
            landmarks={
                "left_shoulder": [0, 0, 0],
                "right_shoulder": [1, 0, 0],
                "left_hip": [2, 0, 0],
            }
        )
    assert body.soma is None
    with pytest.raises(ValueError, match="Unknown SOMA anchor"):
        from patient_digital_twin import AnatomicalStructure, Kind

        body.anatomy.structures["liver"] = AnatomicalStructure(
            "liver", Kind.ORGAN, np.zeros((3, 3)), np.array([[0, 1, 2]])
        )
        body.AttachExternalBody(
            ArticulatedLayer(), body_to_soma=np.eye(4), anchors={"liver": "missing"}
        )
    assert body.soma is None
