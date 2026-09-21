# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("viser")
from test_soma import bound_body as bound_body  # noqa: PLC0414 - pytest fixture export


def load_viewer():
    spec = importlib.util.spec_from_file_location(
        "patient_viewer", Path(__file__).parents[1] / "examples" / "viewer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_viewer_serves_and_updates_rigid_frames(bound_body):
    module = load_viewer()
    viewer = module.HumanBodyViewer(bound_body, port=0)
    try:
        port = viewer.server.get_port()
        with urlopen(f"http://127.0.0.1:{port}", timeout=5) as response:
            assert response.status == 200
            assert b"<html" in response.read().lower()
        for pose in viewer.presets:
            viewer.set_pose(pose)
            np.testing.assert_allclose(
                viewer.skin.vertices, bound_body.soma_body.vertices
            )
            for name, handle in viewer.meshes.items():
                structure = bound_body.structures[name]
                np.testing.assert_allclose(
                    handle.position, structure.local_to_world[:3, 3]
                )
                rotation = Rotation.from_quat(np.asarray(handle.wxyz)[[1, 2, 3, 0]])
                np.testing.assert_allclose(
                    rotation.as_matrix(), structure.local_to_world[:3, :3], atol=1e-7
                )
                np.testing.assert_allclose(
                    rotation.apply(handle.vertices) + handle.position,
                    structure.world_vertices,
                    atol=1e-7,
                )
        result = module.validate_poses(bound_body)
        assert result["passed"]
        assert set(result["poses"]) == {
            "scan",
            "arms_out",
            "seated",
            "torso_bend",
        }
    finally:
        viewer.close()


def test_viewer_respects_configuration_and_reenables_meshes(bound_body):
    module = load_viewer()
    bound_body.set_anatomy_enabled(False)
    viewer = module.HumanBodyViewer(bound_body, port=0)
    try:
        assert viewer.skin.visible
        assert all(not handle.visible for handle in viewer.meshes.values())
        for pose in viewer.presets:
            viewer.set_pose(pose)
            bound_body.set_anatomy_enabled(True)
            viewer.refresh()
            for name, handle in viewer.meshes.items():
                assert handle.visible
                np.testing.assert_allclose(
                    handle.vertices,
                    bound_body.structures[name].vertices,
                    atol=1e-7,
                )
            bound_body.set_structure_enabled("liver", False)
            viewer.refresh()
            assert not viewer.meshes["liver"].visible
            assert viewer.meshes["kidney_left"].visible
            bound_body.set_structure_enabled("liver", True)
            bound_body.set_anatomy_enabled(False)
            viewer.refresh()
            assert all(not handle.visible for handle in viewer.meshes.values())
    finally:
        viewer.close()


def test_viewer_multiple_structure_selection(bound_body):
    viewer = load_viewer().HumanBodyViewer(bound_body, port=0)
    try:

        def visible():
            return {name for name, handle in viewer.meshes.items() if handle.visible}

        assert visible() == {"liver", "kidney_left"}
        viewer.set_visible_structures([])
        assert visible() == set()
        assert viewer.skin.visible
        viewer.set_visible_structures(["liver"])
        for pose in viewer.presets:
            viewer.set_pose(pose)
            assert visible() == {"liver"}
        viewer.set_visible_structures(["liver", "kidney_left"])
        assert visible() == {"liver", "kidney_left"}
        viewer.interior.value = False
        viewer.refresh()
        assert visible() == set()
        viewer.interior.value = True
        viewer.refresh()
        assert visible() == {"liver", "kidney_left"}
        bound_body.set_structure_enabled("liver", False)
        viewer.refresh()
        assert visible() == {"kidney_left"}
        bound_body.set_structure_enabled("liver", True)
        viewer.refresh()
        assert visible() == {"liver", "kidney_left"}
        with pytest.raises(ValueError, match="Unknown mesh structures"):
            viewer.set_visible_structures(["missing"])
        assert all(c.value for c in viewer.structure_controls.values())
    finally:
        viewer.close()
