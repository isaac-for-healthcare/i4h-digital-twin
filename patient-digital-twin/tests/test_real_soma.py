# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in actual SOMALayer regression, using small synthetic interior masks.

These are geometry fixtures, not a claim of fitting patient anatomy.
Run PATIENT_TWIN_TEST_SOMA=1 with local SOMA assets available.
"""

from patient_digital_twin import HumanBody
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PATIENT_TWIN_TEST_SOMA") != "1",
    reason="Set PATIENT_TWIN_TEST_SOMA=1 to load real SOMA assets",
)


def test_real_soma_importer_rigidity_and_containment():
    torch = pytest.importorskip("torch")
    trimesh = pytest.importorskip("trimesh")
    from patient_digital_twin import HumanBody, SegmentationImporter
    from soma import SOMALayer
    from test_viewer import load_viewer

    torch.set_num_threads(4)
    layer = SOMALayer(
        identity_model_type="soma",
        lod="low",
        device="cpu",
        mode="dense",
        correctives_model_path=None,
        enable_procedural_transforms=False,
    )
    reference = HumanBody()
    reference.AttachExternalBody(layer, body_to_soma=np.eye(4))
    skin = trimesh.Trimesh(
        reference.soma.soma_body.vertices, reference.soma.soma_body.faces, process=False
    )
    joints = reference.soma.soma_body.joints
    # Place tiny test masks around actual interior bone-segment midpoints.
    centers = {
        "liver": (joints["Spine1"] + joints["Spine2"]) / 2,
        "humerus_left": (joints["LeftArm"] + joints["LeftForeArm"]) / 2,
        "femur_left": (joints["LeftLeg"] + joints["LeftShin"]) / 2,
    }
    origin = np.floor(np.min(list(centers.values()), axis=0) / 0.004) * 0.004 - 0.04
    high = np.max(list(centers.values()), axis=0) + 0.04
    shape = np.ceil((high - origin) / 0.004).astype(int)[::-1]
    z, y, x = np.indices(shape)
    xyz = np.stack([x, y, z], axis=-1) * 0.004 + origin
    masks = np.zeros(tuple(shape), dtype=np.uint8)
    for index, center in enumerate(centers.values(), 1):
        assert skin.contains([center])[0], (
            "Fixture center must lie inside actual SOMA skin"
        )
        masks[np.linalg.norm(xyz - center, axis=-1) < 0.008] = index
    affine = np.diag([0.004, 0.004, 0.004, 1])
    affine[:3, 3] = origin
    body = HumanBody(SegmentationImporter.from_array(
        masks,
        {i: name for i, name in enumerate(centers, 1)},
        affine_xyz_to_imaging_m=affine,
    ).to_anatomy_collection())
    source_landmarks = {
        key: (joints[name] - body.anatomy.body_to_imaging[:3, 3]).tolist()
        for key, name in {
            "left_shoulder": "LeftArm",
            "right_shoulder": "RightArm",
            "left_hip": "LeftLeg",
            "right_hip": "RightLeg",
        }.items()
    }
    body.AttachExternalBody(layer, landmarks=source_landmarks)
    np.testing.assert_allclose(body.soma.body_to_soma, body.anatomy.body_to_imaging, atol=1e-5)
    vertices = {name: item.vertices.copy() for name, item in body.anatomy.structures.items()}
    viewer = load_viewer()
    report = viewer.validate_poses(body)
    assert report["passed"], report
    for name, item in body.anatomy.structures.items():
        np.testing.assert_array_equal(item.vertices, vertices[name])
