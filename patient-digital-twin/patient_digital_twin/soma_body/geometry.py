# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA output decoding and anatomical joint assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import rigid_transform, to_numpy, transform_points
from ..structures import AnatomicalStructure

SOMA_JOINT_NAMES = {
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_hip": "LeftLeg",
    "right_hip": "RightLeg",
    "left_knee": "LeftShin",
    "right_knee": "RightShin",
}
TRUNK_LANDMARKS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
LIMB_CHAINS = (
    ("LeftArm", "left_shoulder", "left_elbow", "LeftForeArm"),
    ("RightArm", "right_shoulder", "right_elbow", "RightForeArm"),
    ("LeftLeg", "left_hip", "left_knee", "LeftShin"),
    ("RightLeg", "right_hip", "right_knee", "RightShin"),
)


@dataclass(frozen=True)
class PosedBody:
    """Single SOMA evaluation, XYZ meters and public joint world frames."""

    vertices: np.ndarray
    faces: np.ndarray
    joints: dict[str, np.ndarray]
    transforms: dict[str, np.ndarray]

    @classmethod
    def from_output(cls, layer, output):
        """Validate one SOMA forward result and copy its skin and world frames.

        output must contain vertices (1, V, 3) and transforms (1, J, 4, 4),
        including Root. Joint centers come from those frames, not LBS matrices.
        Call this after evaluation, or let SomaRepresentation do it for you.
        """
        vertices, transforms = (
            to_numpy(output["vertices"]),
            to_numpy(output["transforms"]),
        )
        names = list(layer.public_joint_names)
        if vertices.ndim != 3 or vertices.shape[0] != 1 or vertices.shape[2] != 3:
            raise ValueError("HumanBody expects a single SOMA body (batch size 1)")
        if transforms.shape != (1, len(names), 4, 4):
            raise ValueError(
                "SOMA transforms must include all public joints, including Root"
            )
        frames = {
            name: rigid_transform(matrix) for name, matrix in zip(names, transforms[0])
        }
        if not np.isfinite(vertices).all():
            raise ValueError("SOMA returned non-finite skin vertices")
        return cls(
            vertices[0].copy(),
            to_numpy(layer.faces).astype(np.int32),
            {name: matrix[:3, 3] for name, matrix in frames.items()},
            frames,
        )


def default_anchor(structure: AnatomicalStructure, body: PosedBody, alignment) -> str:
    """Anatomical joint assignment; central tissues never attach to nearby hands.

    A structure spanning multiple articulations remains one rigid object.
    Override this mapping for a specific anatomy or segmentation scheme.
    """
    name = structure.name.lower()
    from ..catalog import matching_hints

    regions, side = matching_hints(structure.name)
    if side is None:
        side = (
            "left"
            if name.endswith("_left")
            else "right"
            if name.endswith("_right")
            else None
        )
    prefix = side.title() if side else None
    if prefix:
        for terms, suffix in (
            (("humerus",), "Arm"),
            (("femur",), "Leg"),
            (("tibia", "fibula", "patella"), "Shin"),
            (("radius", "ulna"), "ForeArm"),
            (("clavicula", "scapula"), "Shoulder"),
        ):
            if (
                any(term in name for term in terms)
                and prefix + suffix in body.transforms
            ):
                return prefix + suffix
    if "head" in regions or name in ("brain", "skull"):
        candidates = ["Head"]
    elif "pelvis" in regions and "thorax" not in regions:
        candidates = ["Hips", "Spine1"]
    else:
        candidates = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head"]
        if "thorax" in regions:
            candidates = ["Spine2", "Chest"]
        elif "abdomen" in regions:
            candidates = ["Spine1", "Spine2"]
    candidates = [joint for joint in candidates if joint in body.transforms]
    if not candidates:
        raise ValueError("SOMA layer is missing anatomical anchor joints")
    center = transform_points(
        transform_points(structure.mesh.vertices, structure.local_to_body).mean(0),
        alignment,
    )
    return min(
        candidates, key=lambda joint: np.linalg.norm(center - body.joints[joint])
    )
