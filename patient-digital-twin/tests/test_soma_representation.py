# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct SOMA owner/lifecycle tests with a deterministic CPU model fixture."""

import json

import numpy as np
import pytest
from patient_digital_twin import AnatomyCollection, SomaRepresentation

torch = pytest.importorskip("torch")
pytest.importorskip("trimesh")
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
    body.structures["liver"].vertices += [0.95, 0, 0]
    originals = {name: s.vertices.copy() for name, s in body.structures.items()}
    body.attach_soma(ShapeResponsiveLayer(), **body.soma_configuration)
    assert not body.check_containment()["liver"]["passed"]
    report = body.fit_soma_shape(components=1, iterations=4)
    assert report is body.soma.shape_fit_report
    assert report["identity_coeffs"][0] > 0
    assert report["history"][0]["active_points"] > 0
    assert all(result["passed"] for result in body.check_containment().values())
    for name, structure in body.structures.items():
        np.testing.assert_array_equal(structure.vertices, originals[name])
    assert len(body.soma_layer._forward_hooks) == 1


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
        landmarks=measured,
    )
    baseline = {name: s.body_vertices.copy() for name, s in body.structures.items()}
    body.attach_soma(
        layer, poses=np.zeros((1, 7, 3)), global_scale=1, refine_limbs=False
    )
    np.testing.assert_allclose(body.body_to_soma, alignment, atol=1e-7)
    assert body.landmarks == {} and measured  # Do not clear the caller's dictionary.
    assert body.structures["humerus_left"].anchor_joint == "LeftArm"
    assert body.structures["liver"].anchor_joint in ("Spine1", "Spine2")
    for name, point in measured.items():
        np.testing.assert_allclose(
            transform_points(point, body.body_to_soma),
            body.soma_body.joints[SOMA_JOINT_NAMES[name]],
            atol=1e-7,
        )
    for name, structure in body.structures.items():
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
    body.pose(pose, transl=[[0.1, -0.2, 0.3]])
    for name, structure in body.structures.items():
        np.testing.assert_allclose(structure.body_vertices, baseline[name])
        np.testing.assert_allclose(
            np.linalg.norm(
                structure.world_vertices[:, None] - structure.world_vertices[None, :],
                axis=-1,
            ),
            np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1),
            atol=1e-8,
        )
    config = json.loads(json.dumps(body.soma_configuration))
    expected = {name: s.local_to_anchor.copy() for name, s in body.structures.items()}
    body.attach_soma(layer, **config)  # Explicit stored joints bypass matching.
    for name, structure in body.structures.items():
        np.testing.assert_allclose(structure.local_to_anchor, expected[name])
