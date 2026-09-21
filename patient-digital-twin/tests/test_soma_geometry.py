# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA output decoding is testable using arrays without the SOMA runtime."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from patient_digital_twin import AnatomicalStructure, Kind, PosedBody
from patient_digital_twin.soma_body.geometry import default_anchor


@pytest.fixture
def output_pair():
    layer = SimpleNamespace(
        public_joint_names=["Root", "Hips"], faces=np.array([[0, 1, 2]])
    )
    frames = np.repeat(np.eye(4)[None, None], 2, axis=1)
    frames[0, 1, :3, 3] = [1, 2, 3]
    output = {
        "vertices": np.zeros((1, 3, 3)),
        "transforms": frames,
        "joints": np.zeros((1, 1, 3)),
    }
    return layer, output


def test_posed_body_copies_geometry_and_decodes_root_inclusive_frames(output_pair):
    layer, output = output_pair
    posed = PosedBody.from_output(layer, output)
    assert posed.vertices.shape == (3, 3) and posed.faces.dtype == np.int32
    assert list(posed.transforms) == ["Root", "Hips"]
    np.testing.assert_array_equal(posed.joints["Hips"], [1, 2, 3])
    output["vertices"][:] = 99
    output["transforms"][:] = 99
    layer.faces[:] = 2
    assert not posed.vertices.any()
    np.testing.assert_array_equal(posed.faces, [[0, 1, 2]])
    np.testing.assert_array_equal(posed.transforms["Hips"][:3, 3], [1, 2, 3])
    with pytest.raises(FrozenInstanceError):
        posed.vertices = np.ones((3, 3))


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("vertices", np.zeros((2, 3, 3)), "batch size"),
        ("vertices", np.zeros((3, 3)), "batch size"),
        ("vertices", np.full((1, 3, 3), np.nan), "non-finite"),
        ("transforms", np.eye(4)[None, None], "including Root"),
        (
            "transforms",
            np.repeat(np.diag([2.0, 1.0, 1.0, 1.0])[None, None], 2, axis=1),
            "rigid",
        ),
    ],
)
def test_invalid_soma_outputs_rejected(output_pair, key, value, match):
    layer, output = output_pair
    output[key] = value
    with pytest.raises(ValueError, match=match):
        PosedBody.from_output(layer, output)


@pytest.mark.parametrize(
    "name,kind,region,expected",
    [
        ("humerus_left", Kind.BONE, "upper_limb", "LeftArm"),
        ("femur_left", Kind.BONE, "lower_limb", "LeftLeg"),
        ("tibia_left", Kind.BONE, "lower_limb", "LeftShin"),
        ("radius_left", Kind.BONE, "upper_limb", "LeftForeArm"),
        ("scapula_left", Kind.BONE, "thorax", "LeftShoulder"),
        ("brain", Kind.ORGAN, "head", "Head"),
        ("liver", Kind.ORGAN, "abdomen", "Spine1"),
        ("heart", Kind.ORGAN, "thorax", "Spine2"),
        ("urinary_bladder", Kind.ORGAN, "pelvis", "Hips"),
    ],
)
def test_default_anchor_uses_anatomy_not_nearby_hands(name, kind, region, expected):
    names = [
        "Hips",
        "Spine1",
        "Spine2",
        "Chest",
        "Head",
        "LeftArm",
        "LeftLeg",
        "LeftShin",
        "LeftForeArm",
        "LeftShoulder",
    ]
    frames = {name: np.eye(4) for name in names}
    for index, frame in enumerate(frames.values()):
        frame[0, 3] = index + 1
    frames["LeftArm"][0, 3] = 0  # Nearest skin/joint is deliberately wrong for organs.
    posed = PosedBody(
        np.empty((0, 3)),
        np.empty((0, 3), dtype=int),
        {name: frame[:3, 3] for name, frame in frames.items()},
        frames,
    )
    structure = AnatomicalStructure(
        name,
        kind,
        vertices=np.zeros((3, 3)),
        enabled=False,
    )
    assert default_anchor(structure, posed, np.eye(4)) == expected
    with pytest.raises(ValueError, match="missing"):
        default_anchor(
            structure, PosedBody(posed.vertices, posed.faces, {}, {}), np.eye(4)
        )
