# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browser viewer for a SOMA skin with rigid internal anatomy.

Install with pip install -e '.[soma,viewer]' from patient-digital-twin/. Run:
    python examples/viewer.py --segmentation sample_label.nii.gz \
        --labels /path/to/NV-Generate-CTMR/configs/label_dict.json
Use --validate-only --report report.json for noninteractive pose regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
from patient_digital_twin import Kind, SegmentationImporter
from patient_digital_twin.geometry import to_numpy


def pose_presets(body) -> dict[str, np.ndarray]:
    """Scan pose plus independent arm, hip/knee and spine articulation."""
    base = to_numpy(body.soma_parameters["poses"]).copy()
    names = list(body.soma_layer.public_joint_names)
    # The viewer's scan preset is arms down. The original attachment pose
    # remains internal to SOMA's rigid bindings, not a selectable display pose.
    for joint, angle in {
        "LeftArm": [0, 0, -np.deg2rad(80)],
        "RightArm": [0, 0, np.deg2rad(80)],
        "LeftForeArm": [0, 0, 0],
        "RightForeArm": [0, 0, 0],
    }.items():
        if joint in names:
            base[0, names.index(joint) - 1] = angle
    presets = {"scan": base}
    for name, changes in {
        "arms_out": {"LeftArm": [0, 0, 0], "RightArm": [0, 0, 0]},
        "seated": {
            "LeftLeg": [-np.pi / 2, 0, 0],
            "RightLeg": [-np.pi / 2, 0, 0],
            "LeftShin": [np.pi / 2, 0, 0],
            "RightShin": [np.pi / 2, 0, 0],
        },
        "torso_bend": {"Spine2": [0.2, 0, 0], "Chest": [0.15, 0, 0]},
    }.items():
        pose = base.copy()
        for joint, angle in changes.items():
            if joint in names:
                pose[0, names.index(joint) - 1] = angle
        presets[name] = pose
    return presets


def validate_poses(body, presets=None) -> dict:
    """Check every meshed structure for every pose, restoring the scan pose."""
    presets = pose_presets(body) if presets is None else presets
    reports = {}
    try:
        for name, pose in presets.items():
            body.pose(pose)
            report = body.check_containment()
            failures = [key for key, result in report.items() if not result["passed"]]
            reports[name] = {
                "passed": not failures,
                "failed_structures": failures,
                "structures": report,
            }
            print(
                f"{name}: {len(report) - len(failures)}/{len(report)} structures contained",
                flush=True,
            )
    finally:
        body.pose(pose_presets(body)["scan"])
    return {
        "passed": all(result["passed"] for result in reports.values()),
        "alignment_errors_m": body.alignment_errors_m,
        "shape_fit": body.shape_fit_report,
        "bone_fit": body.bone_fit_report,
        "source_path": body.source_path,
        "soma_configuration": body.soma_configuration,
        "poses": reports,
    }


class HumanBodyViewer:
    """Transparent skin, tissue visibility, named poses, and joint controls."""

    def __init__(self, body, *, host="127.0.0.1", port=8080):
        """Start a Viser server for an already attached HumanBody; call close() to stop.

        Geometry is shared with body. Use set_pose() for named poses or call
        refresh() after programmatic pose/configuration changes. port=0 selects
        an available port, useful for tests.
        """
        import viser
        from scipy.spatial.transform import Rotation

        if body.soma_body is None:
            raise ValueError("Attach SOMA before constructing the viewer")
        self.body, self.presets = body, pose_presets(body)
        self.lock = threading.RLock()
        self.current_pose = self.presets["scan"].copy()
        body.pose(self.current_pose)
        self.server = viser.ViserServer(
            host=host, port=port, label="Patient digital twin"
        )
        self.server.scene.set_up_direction("+y")
        self.skin = self.server.scene.add_mesh_simple(
            "/skin",
            body.soma_body.vertices,
            body.soma_body.faces,
            color=(185, 195, 215),
            opacity=0.22,
            side="double",
        )
        self.meshes = {}
        for name, structure in body.structures.items():
            if structure.mesh.vertices is None:
                continue
            color = (
                (232, 226, 205)
                if structure.kind == Kind.BONE
                else tuple(
                    65 + int(v) // 2 for v in hashlib.sha256(name.encode()).digest()[:3]
                )
            )
            matrix = structure.local_to_world
            xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            self.meshes[name] = self.server.scene.add_mesh_simple(
                "/anatomy/" + name,
                structure.mesh.vertices,
                structure.mesh.faces,
                color=color,
                side="double",
                position=matrix[:3, 3],
                wxyz=xyzw[[3, 0, 1, 2]],
                visible=not structure.is_empty,
            )

        self.preset = self.server.gui.add_dropdown("Pose", tuple(self.presets))
        self.opacity = self.server.gui.add_slider(
            "Skin opacity", min=0, max=1, step=0.01, initial_value=0.22
        )
        self.interior = self.server.gui.add_checkbox("Show anatomy", initial_value=True)
        with self.server.gui.add_folder("Visible structures", expand_by_default=False):
            self.select_all = self.server.gui.add_button("Select all")
            self.clear_selection = self.server.gui.add_button("Clear selection")
            self.structure_controls = {
                name: self.server.gui.add_checkbox(name, initial_value=True)
                for name in self.meshes
            }
        joints = tuple(body.soma_layer.public_joint_names[1:])
        self.joint = self.server.gui.add_dropdown("Joint", joints, initial_value="Hips")
        self.angles = [
            self.server.gui.add_slider(
                f"Axis-angle {axis} (rad)",
                min=-3.15,
                max=3.15,
                step=0.01,
                initial_value=0,
            )
            for axis in "XYZ"
        ]
        self.check = self.server.gui.add_button("Check containment (all meshes)")
        self.status = self.server.gui.add_markdown(
            "Containment has not been checked for this pose."
        )
        self._setting_sliders = False

        @self.preset.on_update
        def _preset(event):
            self.set_pose(self.preset.value)
            self._sync_sliders()

        @self.opacity.on_update
        def _opacity(event):
            self.skin.opacity = self.opacity.value

        @self.interior.on_update
        def _visibility(event):
            with self.lock, self.server.atomic():
                self._update_visibility()

        for control in self.structure_controls.values():
            control.on_update(_visibility)

        @self.select_all.on_click
        def _select_all(event):
            self.set_visible_structures(self.meshes)

        @self.clear_selection.on_click
        def _clear_selection(event):
            self.set_visible_structures(())

        @self.joint.on_update
        def _joint(event):
            self._sync_sliders()

        for control in self.angles:

            @control.on_update
            def _angle(event):
                if self._setting_sliders or event.client is None:
                    return
                with self.lock:
                    index = joints.index(self.joint.value)
                    self.current_pose[0, index] = [c.value for c in self.angles]
                    self.body.pose(self.current_pose)
                    self.refresh()

        @self.check.on_click
        def _check(event):
            with self.lock:
                self.status.content = "Checking every mesh vertex and face center…"
                try:
                    report = self.body.check_containment()
                    failures = [
                        f"{name}: {item['outside_points']} outside, "
                        f"max {1000 * item['max_outside_m']:.1f} mm"
                        for name, item in report.items()
                        if not item["passed"]
                    ]
                    self.status.content = (
                        "All tested points are contained."
                        if not failures
                        else "Containment failures:\n\n" + "\n\n".join(failures)
                    )
                except ValueError as exc:
                    self.status.content = str(exc)

        @self.server.on_client_connect
        def _camera(client):
            vertices = self.body.soma_body.vertices
            center = (vertices.min(0) + vertices.max(0)) / 2
            distance = max(float(np.ptp(vertices, axis=0).max()), 0.5)
            # Viser's position setter also translates look_at: set it first.
            client.camera.position = tuple(
                center + distance * np.array([1.0, 0.2, 1.5])
            )
            client.camera.look_at = tuple(center)

    def _sync_sliders(self):
        """Reflect the selected joint without recursively triggering pose callbacks."""
        self._setting_sliders = True
        try:
            index = (
                list(self.body.soma_layer.public_joint_names).index(self.joint.value)
                - 1
            )
            for control, value in zip(self.angles, self.current_pose[0, index]):
                control.value = float(value)
        finally:
            self._setting_sliders = False

    def set_pose(self, name):
        """Apply a named preset and refresh geometry; containment is not implicit."""
        with self.lock:
            self.status.content = f"Computing pose {name}…"
            self.current_pose = self.presets[name].copy()
            self.body.pose(self.current_pose)
            self.refresh()
            self.status.content = (
                f"Showing pose {name}; containment has not been checked."
            )

    def set_visible_structures(self, names):
        """Select any subset of mesh names; an empty iterable hides all anatomy.

        Selection persists across poses and the Show anatomy toggle. Disabled
        structures remain hidden until enabled in the body's configuration.
        """
        selected = set(names)
        unknown = selected.difference(self.meshes)
        if unknown:
            raise ValueError(f"Unknown mesh structures: {sorted(unknown)}")
        with self.lock, self.server.atomic():
            for name, control in self.structure_controls.items():
                control.value = name in selected
            self._update_visibility()

    def _update_visibility(self):
        """Apply the checklist and master toggle without recalculating geometry."""
        for name, handle in self.meshes.items():
            handle.visible = (
                not self.body.structures[name].is_empty
                and self.interior.value
                and self.structure_controls[name].value
            )

    def refresh(self):
        """Push current skin, transforms, and enabled anatomy into existing scene handles."""
        from scipy.spatial.transform import Rotation

        with self.server.atomic():
            self.skin.vertices = self.body.soma_body.vertices
            self._update_visibility()
            for name, handle in self.meshes.items():
                structure = self.body.structures[name]
                if structure.is_empty:
                    continue
                handle.vertices = structure.vertices
                matrix = structure.local_to_world
                xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
                handle.wxyz = xyzw[[3, 0, 1, 2]]
                handle.position = matrix[:3, 3]
        self.status.content = "Pose changed; containment has not been checked."

    def close(self):
        """Stop the local viewer server and release its network resources."""
        self.server.stop()


def main():
    """CLI entry: import, attach/optionally fit, then serve or validate all presets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument(
        "--anatomy-config", type=Path, help="YAML mesh availability settings"
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        help="JSON attach_soma arguments: poses, identity_coeffs, scale_params, "
        "global_scale, body_to_soma, landmarks, anchors",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lod", choices=("low", "mid"), default="low")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address; use 0.0.0.0 for remote access",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--pose",
        choices=("scan", "arms_out", "seated", "torso_bend"),
        default="scan",
        help="Initial display pose after scan-based binding",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--fit-shape",
        action="store_true",
        help="Fit SOMA identity across all four poses",
    )
    parser.add_argument("--report", type=Path, default=Path("containment.json"))
    parser.add_argument(
        "--save-parameters",
        type=Path,
        help="Save fitted SOMA configuration for --parameters",
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    importer = SegmentationImporter(args.segmentation, args.labels)
    body = importer.to_human_body(configuration=args.anatomy_config)
    params = json.loads(args.parameters.read_text()) if args.parameters else {}
    body.attach_soma(
        device=args.device, lod=args.lod, data_root=args.data_root, **params
    )
    print(
        f"Imported {sum(s.vertices is not None for s in body.structures.values())} meshes."
    )
    print(
        "Landmark residuals (mm):",
        {key: round(1000 * v, 1) for key, v in body.alignment_errors_m.items()},
    )
    if args.fit_shape:
        body.fit_soma_shape(pose_presets(body).values())
        body.fit_bone_anchors(pose_presets(body).values())
    if args.save_parameters:
        args.save_parameters.parent.mkdir(parents=True, exist_ok=True)
        args.save_parameters.write_text(
            json.dumps(body.soma_configuration, indent=2) + "\n"
        )
    if args.validate_only:
        report = validate_poses(body)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(0 if report["passed"] else 1)
    viewer = HumanBodyViewer(body, host=args.host, port=args.port)
    viewer.set_pose(args.pose)
    viewer.preset.value = args.pose
    print(f"Open http://{args.host}:{args.port}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
