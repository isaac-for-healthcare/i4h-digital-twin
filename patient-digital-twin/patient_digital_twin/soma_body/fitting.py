# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA skin fitting and rigid bone attachment optimization."""

from __future__ import annotations

import numpy as np

from ..geometry import to_numpy, transform_points
from ..structures import Kind
from .geometry import PosedBody


def is_bone(structure):
    """Include composite bone masks in rigid bone fitting."""
    return structure.kind == Kind.BONE or structure.name in ("skull", "vertebrae")


def skin_mesh(posed):
    """Build a closed outward SOMA skin for fitting diagnostics."""
    import trimesh

    skin = trimesh.Trimesh(posed.vertices, posed.faces, process=False)
    if not skin.is_watertight or not skin.is_winding_consistent or skin.volume <= 0:
        raise ValueError("Fitting needs closed, consistently outward-wound SOMA skin")
    return skin


def fit_soma_identity(body, poses, *, components=48, iterations=12, margin_m=0.003):
    """Fit bounded SOMA PCA coefficients to the anatomy envelope in several poses.

    body is a SomaRepresentation; application code should call
    SomaRepresentation.fit_soma_shape().
    Source geometry and bone sizes stay fixed. Each candidate identity rebinds
    the original body-frame geometry at the scan pose. Active surface constraints
    move SOMA skin outward around anatomy; bounded steps limit identity changes.
    All anatomy is constrained in every pose and moves rigidly. Coefficients
    stay within +/-3
    standard deviations. This is not a clinical registration algorithm.
    """
    import torch
    import trimesh
    from scipy.optimize import linprog

    params = body.soma_parameters
    layer = body.soma_layer
    reference = params["identity_coeffs"]
    coeffs = to_numpy(reference)[0].copy()
    count = min(components, len(coeffs))
    if count < 1 or iterations < 1 or margin_m <= 0:
        raise ValueError("Positive components, iterations and margin are required")
    pose_list = [params["poses"]] + [
        torch.as_tensor(p, dtype=reference.dtype, device=reference.device)
        for p in poses
        if not np.allclose(to_numpy(p), to_numpy(params["poses"]))
    ]
    structures = [s for s in body.structures.values() if s.mesh.vertices is not None]
    samples = {}
    for s in structures:
        # Deterministic fitting samples; acceptance uses every vertex/face center.
        stride = max(1, len(s.mesh.vertices) // 2500)
        samples[s.name] = s.mesh.vertices[::stride]

    def evaluate(candidate):
        call = dict(params)
        call["identity_coeffs"] = torch.as_tensor(
            candidate[None], dtype=reference.dtype, device=reference.device
        )
        evaluated = []
        with torch.no_grad():
            for pose in pose_list:
                call["poses"] = pose
                evaluated.append(PosedBody.from_output(layer, layer.forward(**call)))
        scan_inverse = {
            key: np.linalg.inv(value) for key, value in evaluated[0].transforms.items()
        }
        point_sets = []
        for posed in evaluated:
            points = []
            for structure in structures:
                matrix = (
                    posed.transforms[structure.anchor_joint]
                    @ scan_inverse[structure.anchor_joint]
                    @ body.body_to_soma
                    @ structure.local_to_body
                )
                points.append(transform_points(samples[structure.name], matrix))
            point_sets.append(np.concatenate(points) if points else np.empty((0, 3)))
        return evaluated, point_sets

    history = []

    def maximum_protrusion(candidate):
        evaluated, targets = evaluate(candidate)
        maximum = 0.0
        for posed, target in zip(evaluated, targets):
            skin = skin_mesh(posed)
            outside = target[~skin.contains(target)]
            for start in range(0, len(outside), 4096):
                _, distance, _ = trimesh.proximity.closest_point(
                    skin, outside[start : start + 4096]
                )
                maximum = max(maximum, float(distance.max(initial=0)))
        return maximum

    for iteration in range(iterations):
        evaluated, points = evaluate(coeffs)
        constraints = []
        for posed, target in zip(evaluated, points):
            if not len(target):
                constraints.append(None)
                continue
            skin = skin_mesh(posed)
            # Only nearby/exterior samples need expensive closest-point queries.
            inside = skin.contains(target)
            batches = [
                trimesh.proximity.closest_point(skin, target[start : start + 4096])
                for start in range(0, len(target), 4096)
            ]
            closest, distances, triangles = [
                np.concatenate(items) for items in zip(*batches)
            ]
            gap = np.where(inside, -distances, distances) + margin_m
            active = np.flatnonzero(gap > 0)
            if not len(active):
                constraints.append(None)
                continue
            face = triangles[active]
            bary = trimesh.triangles.points_to_barycentric(
                skin.triangles[face], closest[active]
            )
            constraints.append(
                (active, skin.faces[face], bary, skin.face_normals[face], gap[active])
            )
        gaps = [item[4] for item in constraints if item is not None]
        maximum = max((float(g.max()) for g in gaps), default=0.0)
        history.append(
            {
                "iteration": iteration,
                "max_constraint_m": maximum,
                "active_points": sum(len(g) for g in gaps),
            }
        )
        if maximum < 0.0002:
            break
        columns = []
        epsilon = 0.04
        for component in range(count):
            perturbed = coeffs.copy()
            perturbed[component] += epsilon
            variants, moving_points = evaluate(perturbed)
            values = []
            for base, variant, old_points, new_points, item in zip(
                evaluated, variants, points, moving_points, constraints
            ):
                if item is None:
                    continue
                active, faces, bary, normals, gap = item
                surface_delta = np.einsum(
                    "vk,vkj->vj", bary, variant.vertices[faces] - base.vertices[faces]
                )
                relative = surface_delta - (new_points[active] - old_points[active])
                values.append(np.einsum("vi,vi->v", relative, normals) / epsilon)
            columns.append(np.concatenate(values))
        jacobian = np.stack(columns, axis=1)
        # Minimize the worst protrusion rather than trading a small bone against
        # many organ samples in a mean-squared loss. A bounded step and nonlinear
        # line search prevent PCA fitting from increasing unseen contacts.
        lower = np.maximum(-0.4, -3.0 - coeffs[:count])
        upper = np.minimum(0.4, 3.0 - coeffs[:count])
        solution = linprog(
            np.r_[np.zeros(count), 1.0],
            A_ub=np.column_stack([-jacobian, -np.ones(len(jacobian))]),
            b_ub=-np.concatenate(gaps),
            bounds=[*zip(lower, upper), (0, None)],
            method="highs",
        )
        if not solution.success:
            raise RuntimeError(f"SOMA identity fit failed: {solution.message}")
        delta = solution.x[:count]
        accepted = False
        previous = max(0.0, maximum - margin_m)
        for fraction in (1.0, 0.5, 0.25, 0.125):
            candidate = coeffs.copy()
            candidate[:count] += fraction * delta
            if maximum_protrusion(candidate) < previous - 1e-6:
                coeffs = candidate
                accepted = True
                break
        print(
            f"SOMA shape fit {iteration + 1}: {maximum * 1000:.1f} mm envelope constraint",
            flush=True,
        )
        if not accepted or np.linalg.norm(delta) < 1e-4:
            break
    body.attach_soma(
        layer,
        body_to_soma=body.body_to_soma,
        anchors={s.name: s.anchor_joint for s in structures},
        poses=params["poses"],
        identity_coeffs=coeffs[None],
        scale_params=params["scale_params"],
        global_scale=params["global_scale"],
        refine_limbs=False,
    )
    body.shape_fit_report = {"history": history, "identity_coeffs": coeffs.tolist()}
    return body.shape_fit_report


def fit_rigid_bone_anchors(
    body,
    poses,
    *,
    max_translation_m=0.03,
    max_rotation_deg=25.0,
    margin_m=0.001,
    iterations=16,
):
    """Register rigid bone offsets inside skin over specified poses.

    Only a fixed rigid offset in each bone's joint frame is fitted. Vertex
    coordinates, dimensions, topology, and per-pose joint rotations are never
    changed. Translation/rotation are bounded and recorded for review.
    Failure to find an enclosing rigid placement is reported, not deformed.
    """
    import torch
    import trimesh
    from scipy.optimize import linprog
    from scipy.spatial.transform import Rotation

    params = body.soma_parameters
    layer = body.soma_layer
    reference = params["poses"]
    pose_list = [reference] + [
        torch.as_tensor(p, dtype=reference.dtype, device=reference.device)
        for p in poses
        if not np.allclose(to_numpy(p), to_numpy(reference))
    ]
    evaluated = []
    with torch.no_grad():
        for pose in pose_list:
            call = dict(params)
            call["poses"] = pose
            evaluated.append(PosedBody.from_output(layer, layer.forward(**call)))
    skins = [skin_mesh(posed) for posed in evaluated]
    fitted = {}
    for name, structure in body.structures.items():
        if structure.mesh.vertices is None or not is_bone(structure):
            continue
        vertices = np.concatenate(
            [
                structure.mesh.vertices,
                structure.mesh.vertices[structure.mesh.faces].mean(1),
            ]
        )
        original = structure.local_to_anchor.copy()
        offset = np.zeros(3)
        rotation = np.eye(3)
        matrices = [p.transforms[structure.anchor_joint] for p in evaluated]
        relative = vertices @ original[:3, :3].T
        contact_skins = skins
        if name.startswith("scapula"):
            # Across the armpit gap, the nearest face can belong to an arm.
            # A shoulder blade instead belongs under the trunk skin.
            names = list(layer.public_joint_names)
            arm_ids = [
                i
                for i, joint in enumerate(names)
                if any(
                    joint.startswith(prefix)
                    for prefix in (
                        "LeftArm",
                        "RightArm",
                        "LeftForeArm",
                        "RightForeArm",
                        "LeftHand",
                        "RightHand",
                    )
                )
            ]
            weights = to_numpy(layer.public_skinning_weights())
            arm_weight = weights[:, arm_ids].sum(1)
            keep = arm_weight[evaluated[0].faces].mean(1) < 0.1
            contact_skins = [
                trimesh.Trimesh(p.vertices, p.faces[keep], process=False)
                for p in evaluated
            ]

        def contacts(
            delta,
            orient,
            relative=relative,
            contact_skins=contact_skins,
            matrices=matrices,
            original=original,
        ):
            rows, gaps = [], []
            worst = 0.0
            rotated = relative @ orient.T
            for skin, contact_skin, matrix in zip(skins, contact_skins, matrices):
                points = transform_points(rotated + original[:3, 3] + delta, matrix)
                exterior = ~skin.contains(points)
                outside = points[exterior]
                relative_outside = rotated[exterior]
                for start in range(0, len(outside), 4096):
                    _, distance, triangles = trimesh.proximity.closest_point(
                        contact_skin, outside[start : start + 4096]
                    )
                    normals = contact_skin.face_normals[triangles]
                    local_normals = normals @ matrix[:3, :3]
                    rows.append(
                        np.column_stack(
                            [
                                local_normals,
                                np.cross(
                                    relative_outside[start : start + 4096],
                                    local_normals,
                                ),
                            ]
                        )
                    )
                    gaps.append(distance + margin_m)
                    worst = max(worst, float(distance.max(initial=0)))
            return rows, gaps, worst

        def actual_protrusion(
            delta, orient, relative=relative, matrices=matrices, original=original
        ):
            maximum = 0.0
            for skin, matrix in zip(skins, matrices):
                points = transform_points(
                    relative @ orient.T + original[:3, 3] + delta, matrix
                )
                outside = points[~skin.contains(points)]
                for start in range(0, len(outside), 4096):
                    _, distances, _ = trimesh.proximity.closest_point(
                        skin, outside[start : start + 4096]
                    )
                    maximum = max(maximum, float(distances.max(initial=0)))
            return maximum

        initial_rows, _, initial_constraint = contacts(offset, rotation)
        if not initial_rows:
            continue
        initial = actual_protrusion(offset, rotation)
        for _ in range(iterations):
            rows, gaps, maximum = contacts(offset, rotation)
            if not rows:
                break
            rows, gaps = np.concatenate(rows), np.concatenate(gaps)
            # Linearized rigid twist; minimize worst penetration plus a small
            # L1 motion cost. Rotation is about the bone mesh's local origin.
            constraints = np.zeros((len(rows) + 12, 13))
            constraints[: len(rows), :6] = rows
            constraints[: len(rows), 6] = -1
            constraints[len(rows) : len(rows) + 6, :6] = np.eye(6)
            constraints[len(rows) : len(rows) + 6, 7:] = -np.eye(6)
            constraints[len(rows) + 6 :, :6] = -np.eye(6)
            constraints[len(rows) + 6 :, 7:] = -np.eye(6)
            bounds = list(
                zip(
                    np.maximum(-0.01, -max_translation_m - offset),
                    np.minimum(0.01, max_translation_m - offset),
                )
            )
            solution = linprog(
                [0] * 6 + [1, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001],
                A_ub=constraints,
                b_ub=np.r_[-gaps, np.zeros(12)],
                bounds=bounds + [(-0.12, 0.12)] * 3 + [(0, None)] * 7,
                method="highs",
            )
            if not solution.success:
                break
            candidate = solution.x[:6]
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125):
                trial = offset + fraction * candidate[:3]
                trial_rotation = (
                    Rotation.from_rotvec(fraction * candidate[3:]).as_matrix()
                    @ rotation
                )
                if np.linalg.norm(
                    Rotation.from_matrix(trial_rotation).as_rotvec()
                ) > np.deg2rad(max_rotation_deg):
                    continue
                _, _, after = contacts(trial, trial_rotation)
                if after < maximum - 1e-7:
                    offset, rotation, accepted = trial, trial_rotation, True
                    break
            if not accepted:
                break
        final = actual_protrusion(offset, rotation)
        structure.local_to_anchor[:3, 3] = original[:3, 3] + offset
        structure.local_to_anchor[:3, :3] = rotation @ original[:3, :3]
        fitted[name] = {
            "joint_translation_m": offset.tolist(),
            "joint_rotation_rad": Rotation.from_matrix(rotation).as_rotvec().tolist(),
            "initial_protrusion_m": initial,
            "initial_contact_distance_m": initial_constraint,
            "final_protrusion_m": final,
        }
        print(
            f"Rigid {name}: {initial * 1000:.2f} -> {final * 1000:.2f} mm protrusion",
            flush=True,
        )
    body.bone_fit_report = fitted
    body.pose()
    return fitted
