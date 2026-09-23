# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SOMA-X attachment, articulation, fitting, and containment diagnostics.

All model-specific state lives here; HumanBody is the application facade.
Bindings include retained disabled meshes so re-enabling is
instantaneous at the current pose. Containment only reports enabled anatomy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..geometry import (
    rigid_transform,
    rotation_between,
    solve_similarity,
    to_numpy,
    transform_points,
)
from .geometry import (
    LIMB_CHAINS,
    SOMA_JOINT_NAMES,
    TRUNK_LANDMARKS,
    PosedBody,
    default_anchor,
)

if TYPE_CHECKING:
    from soma import SOMALayer

    from ..anatomy import AnatomyCollection


@dataclass
class SomaRepresentation:
    """Own one patient\'s model, fitted parameters, bindings and latest pose.

    Usually accessed as body.soma. Construct directly with an AnatomyCollection
    for advanced use, then call attach_soma() before evaluating or fitting.
    Optional model dependencies are imported only when those methods are used.
    """

    anatomy: AnatomyCollection
    soma_layer: SOMALayer | None = field(default=None, repr=False)
    landmarks: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    body_to_soma: np.ndarray = field(default_factory=lambda: np.eye(4), repr=False)
    soma_body: PosedBody | None = field(default=None, init=False, repr=False)
    alignment_errors_m: dict[str, float] = field(default_factory=dict, init=False)
    _soma_hook: object = field(default=None, init=False, repr=False)
    _pose_parameters: dict = field(default_factory=dict, init=False, repr=False)
    shape_fit_report: dict = field(default_factory=dict, init=False)
    bone_fit_report: dict = field(default_factory=dict, init=False)

    @property
    def structures(self):
        """Anatomical structures, including temporarily disabled mesh storage."""
        return self.anatomy.structures

    def extract_landmarks(self) -> dict[str, np.ndarray]:
        """Estimate joints from retained humerus/femur surfaces in the body frame.

        Bone principal-axis endpoints approximate shoulders/elbows and hips/knees.
        Spine geometry identifies the proximal end. Source scan bounds, when
        available, exclude cropped endpoints. Visibility does not affect fitting.
        Custom meshes must share a correctly oriented and registered body frame.
        """
        points_by_name = {
            name: transform_points(structure.mesh.vertices, structure.local_to_body)
            for name, structure in self.structures.items()
            if structure.mesh.vertices is not None and len(structure.mesh.vertices)
        }
        if not points_by_name:
            return {}
        spine = [
            points.mean(0)
            for name, points in points_by_name.items()
            if name.startswith("vertebrae") or name in ("sacrum", "spinal_cord")
        ]
        reference = np.mean(
            spine or [p.mean(0) for p in points_by_name.values()], axis=0
        )
        landmarks = {}
        for name in ("humerus_left", "humerus_right", "femur_left", "femur_right"):
            points = points_by_name.get(name)
            if points is None or len(points) < 3 or not np.isfinite(points).all():
                continue
            centered = points - points.mean(0)
            _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
            if singular[0] <= 1e-10:
                continue
            offsets = centered @ vectors[0]
            ends = []
            for selection in (
                offsets <= np.quantile(offsets, 0.12),
                offsets >= np.quantile(offsets, 0.88),
            ):
                endpoint = points[selection]
                cropped = False
                if (
                    self.anatomy.body_to_voxel is not None
                    and self.anatomy.source_shape_xyz is not None
                ):
                    voxels = transform_points(endpoint, self.anatomy.body_to_voxel)
                    cropped = bool(
                        np.any(voxels <= 0)
                        or np.any(
                            voxels >= np.asarray(self.anatomy.source_shape_xyz) - 1
                        )
                    )
                ends.append((endpoint.mean(0), cropped))
            ends.sort(key=lambda end: np.linalg.norm(end[0] - reference))
            joints = (
                ("shoulder", "elbow") if name.startswith("humerus") else ("hip", "knee")
            )
            for joint, (point, cropped) in zip(joints, ends):
                if not cropped:
                    landmarks[f"{name.split('_')[-1]}_{joint}"] = point
        return landmarks

    def attach_soma(
        self,
        layer: SOMALayer | None = None,
        *,
        landmarks=None,
        body_to_soma=None,
        anchors: Mapping[str, str] | None = None,
        anchor_transforms: Mapping[str, np.ndarray] | None = None,
        poses=None,
        identity_coeffs=None,
        scale_params=None,
        global_scale: float | None = None,
        refine_limbs: bool = True,
        arm_abduction_deg: float = 10.0,
        device: str = "cpu",
        lod: str = "low",
        data_root=None,
    ) -> PosedBody:
        """Fit and attach SOMA-X, keeping all anatomy rigid and in meters.

        Automatic alignment uses >=3 non-collinear shoulder/hip landmarks.
        Their coordinates are body-frame XYZ meters. Uniform size fitting
        changes SOMA's global scale, never the patient's mesh geometry. Pass a
        rigid ``body_to_soma`` matrix for a manually registered/partial scan.
        ``poses`` is an optional (1, J-1, 3) axis-angle scan pose. Bone-length
        refinement is only automatic when that pose is not supplied.
        Successful attachment discards stored landmarks. Further poses use
        only the selected anchor joints and their rigid offsets. Reattachment
        needs new landmarks or the saved body_to_soma registration.

        Direct ``body.soma.soma_layer(...)`` calls synchronize anatomy via a forward
        hook. For SOMA's two-phase API, call ``update_from_soma(layer.pose(...))``.
        Reattach after changing identity/lengths if new anchor offsets are needed.
        """
        # Validate available matches before importing/loading the optional model.
        measured = landmarks
        if measured is None:
            measured = self.landmarks or (
                self.extract_landmarks() if body_to_soma is None else {}
            )
        measured = {
            name: np.asarray(point, dtype=float) for name, point in measured.items()
        }
        if any(
            point.shape != (3,) or not np.isfinite(point).all()
            for point in measured.values()
        ):
            raise ValueError("Landmarks must be finite XYZ points in meters")
        common = [name for name in TRUNK_LANDMARKS if name in measured]
        if body_to_soma is None:
            if len(common) < 3:
                raise ValueError(
                    "Alignment needs at least 3 shoulder/hip matches from humerus/femur anatomy; "
                    "import the corresponding bones or provide explicit landmarks or body_to_soma registration"
                )
            source = np.array([measured[name] for name in common])
            if np.linalg.matrix_rank(source - source.mean(0), tol=1e-8) < 2:
                raise ValueError("Alignment needs non-collinear shoulder/hip matches")
        else:
            rigid_transform(body_to_soma)

        import torch
        from scipy.spatial.transform import Rotation

        if layer is None:
            from soma import SOMALayer

            layer = SOMALayer(
                data_root=data_root,
                identity_model_type="soma",
                lod=lod,
                device=device,
                mode="dense" if device == "cpu" else "warp",
                correctives_model_path=None,
                enable_procedural_transforms=False,
            )
        unit = getattr(layer, "output_unit", "m")
        if getattr(unit, "value", unit) not in (1.0, "m", "meters"):
            raise ValueError("SOMA must use output_unit=Unit.METERS")
        layer.eval()
        parameter = next(layer.buffers())
        device = parameter.device
        names = list(layer.public_joint_names)
        missing = {
            SOMA_JOINT_NAMES[name] for name in measured if name in SOMA_JOINT_NAMES
        } - set(names)
        if missing:
            raise ValueError(
                f"SOMA model lacks matching joint anchors: {sorted(missing)}"
            )
        dtype = torch.float32

        def tensor(value):
            return torch.as_tensor(value, dtype=dtype, device=device)

        automatic_pose = poses is None
        if poses is None:
            poses = np.zeros((1, len(names) - 1, 3))
            angle = np.deg2rad(90.0 - arm_abduction_deg)
            poses[0, names.index("LeftArm") - 1, 2] = -angle
            poses[0, names.index("RightArm") - 1, 2] = angle
        params = {
            "poses": tensor(poses),
            "identity_coeffs": tensor(identity_coeffs)
            if identity_coeffs is not None
            else torch.zeros(
                (1, layer.num_shape_components), dtype=dtype, device=device
            ),
            "scale_params": tensor(scale_params)
            if scale_params is not None
            else torch.ones((1, layer.num_scale_params), dtype=dtype, device=device),
            "transl": torch.zeros((1, 3), dtype=dtype, device=device),
            "global_scale": 1.0 if global_scale is None else global_scale,
            "apply_correctives": False,
        }

        def evaluate():
            with torch.no_grad():
                # Avoid updating existing bindings while a refit is in progress.
                return PosedBody.from_output(layer, layer.forward(**params))

        posed = evaluate()
        if body_to_soma is None:
            source = np.array([measured[key] for key in common])
            target = np.array([posed.joints[SOMA_JOINT_NAMES[key]] for key in common])
            if global_scale is None:
                fit = solve_similarity(source, target)
                params["global_scale"] = 1.0 / fit.scale
                posed = evaluate()
                target = np.array(
                    [posed.joints[SOMA_JOINT_NAMES[key]] for key in common]
                )
            alignment = solve_similarity(source, target, fit_scale=False).as_4x4()
        else:
            alignment = rigid_transform(body_to_soma)

        if refine_limbs and automatic_pose and body_to_soma is None:
            # Match actual scan limb directions, including raised arms. The
            # reference script's 50-degree cutoff leaves those scans misaligned.
            for joint, proximal, distal, scale_name in LIMB_CHAINS:
                if proximal not in measured or distal not in measured:
                    continue
                direction = alignment[:3, :3] @ (measured[distal] - measured[proximal])
                length = np.linalg.norm(direction)
                child = SOMA_JOINT_NAMES[distal]
                current_direction = posed.joints[child] - posed.joints[joint]
                current_length = np.linalg.norm(current_direction)
                if min(length, current_length) < 1e-8:
                    raise ValueError(f"Degenerate limb landmarks for {joint}")
                # The neutral trunk has identity rotation, so this world-space
                # correction composes with the existing local limb rotation.
                correction = Rotation.from_rotvec(
                    rotation_between(current_direction, direction)
                )
                index = names.index(joint) - 1
                old = Rotation.from_rotvec(to_numpy(params["poses"][0, index]))
                params["poses"][0, index] = tensor((correction * old).as_rotvec())
                if scale_name in layer.scale_param_names:
                    params["scale_params"][
                        0, list(layer.scale_param_names).index(scale_name)
                    ] *= length / current_length
            posed = evaluate()

        bindings = {}
        for name, structure in self.structures.items():
            if structure.mesh.vertices is None:
                continue
            joint = (
                anchors[name]
                if anchors and name in anchors
                else default_anchor(structure, posed, alignment)
            )
            if joint not in posed.transforms:
                raise ValueError(f"Unknown SOMA anchor {joint!r} for {name}")
            local_to_anchor = (
                np.linalg.inv(posed.transforms[joint])
                @ alignment
                @ structure.local_to_body
            )
            if anchor_transforms and name in anchor_transforms:
                local_to_anchor = anchor_transforms[name]
            bindings[name] = (joint, rigid_transform(local_to_anchor))
        if self._soma_hook is not None:
            self._soma_hook.remove()
        self.soma_layer = layer
        self.body_to_soma = alignment
        self.soma_body = posed
        self._pose_parameters = params
        self.landmarks = {}  # Matching inputs are not needed after successful binding.
        self.alignment_errors_m = {
            key: float(
                np.linalg.norm(
                    transform_points(np.asarray(measured[key]), alignment)
                    - posed.joints[SOMA_JOINT_NAMES[key]]
                )
            )
            for key in SOMA_JOINT_NAMES
            if key in measured
        }
        for name, (joint, offset) in bindings.items():
            structure = self.structures[name]
            structure.anchor_joint, structure.local_to_anchor = joint, offset
            structure.local_to_world = posed.transforms[joint] @ offset
        self._soma_hook = layer.register_forward_hook(
            lambda module, inputs, output: self.update_from_soma(output)
        )
        return self.pose(self.scan_pose)

    @property
    def scan_pose(self) -> np.ndarray:
        """Arms-down presentation pose; retain the imaging pose for registration."""
        if self.soma_layer is None:
            raise RuntimeError("Call attach_soma first")
        pose = to_numpy(self._pose_parameters["poses"]).copy()
        names = list(self.soma_layer.public_joint_names)
        for joint, angle in {
            "LeftArm": [0, 0, -np.deg2rad(80)],
            "RightArm": [0, 0, np.deg2rad(80)],
            "LeftForeArm": [0, 0, 0],
            "RightForeArm": [0, 0, 0],
        }.items():
            if joint in names:
                pose[0, names.index(joint) - 1] = angle
        return pose

    def update_from_soma(self, output) -> None:
        """Synchronize a single SOMA output; use world FK, not LBS matrices."""
        if self.soma_layer is None:
            raise RuntimeError("Call attach_soma first")
        posed = PosedBody.from_output(self.soma_layer, output)
        pending = {}
        for name, structure in self.structures.items():
            if structure.local_to_anchor is not None:
                pending[name] = rigid_transform(
                    posed.transforms[structure.anchor_joint] @ structure.local_to_anchor
                )
        self.soma_body = posed
        for name, matrix in pending.items():
            self.structures[name].local_to_world = matrix

    def fit_soma_shape(self, poses=(), **options) -> dict:
        """Fit SOMA identity to the scan envelope while preserving rigid bone sizes.

        Supply additional pose tensors to constrain bones across articulation.
        Coefficients are bounded to +/-3 standard deviations. The fitting report
        is diagnostic; call check_containment on each required pose afterwards.
        """
        from .fitting import fit_soma_identity

        if self.soma_layer is None:
            raise RuntimeError("Call attach_soma first")
        return fit_soma_identity(self, poses, **options)

    def fit_bone_anchors(self, poses=(), **options) -> dict:
        """Fit bounded rigid bone registration offsets across required poses.

        Offsets remain fixed in the joint frame and are recorded in
        bone_fit_report. No bone is resized, bent, or projected onto the skin.
        """
        from .fitting import fit_rigid_bone_anchors

        if self.soma_layer is None:
            raise RuntimeError("Call attach_soma first")
        return fit_rigid_bone_anchors(self, poses, **options)

    def pose(self, poses=None, *, transl=None, **parameters) -> PosedBody:
        """Evaluate a pose, defaulting to arms down with fitted identity and lengths.

        Explicit poses replace the whole axis-angle tensor. Additional SOMA
        forward parameters (e.g. identity_coeffs) may be supplied as keywords.
        """
        import torch

        if self.soma_layer is None:
            raise RuntimeError("Call attach_soma first")
        params = dict(self._pose_parameters)
        params["poses"] = self.scan_pose
        params.update(parameters)
        for key, value in (("poses", poses), ("transl", transl)):
            if value is not None:
                params[key] = value
        reference = self._pose_parameters["poses"]
        for key in ("poses", "transl", "identity_coeffs", "scale_params"):
            if params.get(key) is not None:
                params[key] = torch.as_tensor(
                    params[key], device=reference.device, dtype=reference.dtype
                )
        with torch.no_grad():
            self.soma_layer(**params)  # Forward hook updates every mesh atomically.
        return self.soma_body

    @property
    def soma_parameters(self) -> dict:
        """Independent copies of the fitted scan-pose SOMA forward parameters."""
        return {
            key: value.clone() if hasattr(value, "clone") else value
            for key, value in self._pose_parameters.items()
        }

    @property
    def soma_configuration(self) -> dict:
        """JSON-ready attachment arguments, including fitted rigid bone offsets."""
        if self.soma_layer is None:
            raise RuntimeError("Call attach_soma first")
        config = {
            key: to_numpy(self._pose_parameters[key]).tolist()
            for key in ("poses", "identity_coeffs", "scale_params", "global_scale")
        }
        config["body_to_soma"] = self.body_to_soma.tolist()
        config["anchors"] = {
            name: s.anchor_joint
            for name, s in self.structures.items()
            if s.local_to_anchor is not None
        }
        config["anchor_transforms"] = {
            name: s.local_to_anchor.tolist()
            for name, s in self.structures.items()
            if s.local_to_anchor is not None
        }
        return config

    def check_containment(
        self, *, tolerance_m: float = 1e-5, include_face_centers: bool = True
    ) -> dict[str, dict]:
        """Test every vertex and (by default) face center against the actual skin.

        No subsampling or convex-hull substitution. This detects protrusions,
        but finite point tests are not a proof against triangle/skin crossings.
        Raises on a non-watertight skin instead of reporting a false pass.
        """
        import trimesh

        if self.soma_body is None:
            raise RuntimeError("Call attach_soma first")
        if not np.isfinite(tolerance_m) or tolerance_m < 0:
            raise ValueError("tolerance_m must be finite and non-negative")
        skin = trimesh.Trimesh(
            self.soma_body.vertices, self.soma_body.faces, process=False
        )
        if not skin.is_watertight or not skin.is_winding_consistent:
            raise ValueError("SOMA skin must be watertight with consistent winding")
        report = {}
        for name, structure in self.structures.items():
            points = structure.world_vertices
            if points is None:
                continue
            if include_face_centers and structure.faces is not None:
                points = np.concatenate([points, points[structure.faces].mean(axis=1)])
            outside_count, maximum = 0, 0.0
            for start in range(0, len(points), 2048):
                chunk = points[start : start + 2048]
                outside = chunk[~skin.contains(chunk)]
                if len(outside):
                    _, distances, _ = trimesh.proximity.closest_point(skin, outside)
                    outside_count += int(np.count_nonzero(distances > tolerance_m))
                    maximum = max(maximum, float(distances.max()))
            report[name] = {
                "points_tested": len(points),
                "outside_points": outside_count,
                "max_outside_m": maximum,
                "passed": outside_count == 0,
            }
        return report
