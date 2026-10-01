# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Topology selection, local-frame storage, and VTK skeleton contracts."""

import numpy as np
import pytest
from patient_digital_twin import (
    AnatomicalStructure,
    CenterlineGraph,
    HumanBody,
    Kind,
    topology,
)
from patient_digital_twin.catalog import CATALOG


def graph():
    return CenterlineGraph(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
        np.array([0.01, 0.01]),
        np.array([[0, 1]]),
    )


def test_body_extracts_tubular_anatomy_including_disabled_and_composite(monkeypatch):
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    names = {
        "aorta": Kind.VESSEL,
        "trachea": CATALOG["trachea"],
        "bronchus": Kind.AIRWAY,
        "portal_vein_and_splenic_vein": Kind.GROUP,
        "liver": Kind.ORGAN,
        "lung_left": Kind.ORGAN,
    }
    body = HumanBody(
        {
            n: AnatomicalStructure(n, k, vertices.copy(), np.array([[0, 1, 2]]))
            for n, k in names.items()
        }
    )
    body.anatomy.structures["aorta"].enabled = False
    body.anatomy.structures["empty"] = AnatomicalStructure("empty", Kind.VESSEL)
    calls = []

    def extract(v, f, **grid):
        calls.append((v, f))
        return graph()

    monkeypatch.setattr(topology, "extract_centerlines", extract)
    result = body.extract_topology()
    assert set(result) == {
        "aorta",
        "trachea",
        "bronchus",
        "portal_vein_and_splenic_vein",
    }
    assert len(calls) == 4
    structure = body.anatomy.structures["aorta"]
    assert structure.centerline is result["aorta"]
    structure.local_to_world[:3, 3] = 1
    np.testing.assert_array_equal(structure.centerline.points, graph().points)
    structure.enabled = True
    assert structure.centerline is result["aorta"]
    structure.vertices = vertices.copy()
    assert structure.centerline is None
    structure.centerline = graph()
    structure.faces = np.array([[0, 2, 1]])
    assert structure.centerline is None


def test_failure_is_named_and_does_not_commit_partial_results(monkeypatch):
    body = HumanBody(
        {
            n: AnatomicalStructure(
                n,
                Kind.VESSEL,
                np.ones((3, 3)),
                np.array([[0, 1, 2]]),
                centerline=graph(),
            )
            for n in ["first", "second"]
        }
    )
    previous = body.anatomy.structures["first"].centerline

    def extract(v, f, **grid):
        if v is body.anatomy.structures["second"].mesh.vertices:
            raise ValueError("bad tube")
        return graph()

    monkeypatch.setattr(topology, "extract_centerlines", extract)
    with pytest.raises(RuntimeError, match="second: bad tube"):
        body.extract_topology()
    assert body.anatomy.structures["first"].centerline is previous


def test_empty_body_and_invalid_spacing():
    assert HumanBody().extract_topology() == {}
    with pytest.raises(ValueError, match="spacing_m"):
        HumanBody().extract_topology(spacing_m=0)


def test_mesh_skeleton_grid_round_trip_and_physical_radii():
    pytest.importorskip("vtk")
    from patient_digital_twin.imaging_to_mesh import mask_to_mesh
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    mask = np.zeros((24, 17, 17), dtype=bool)
    mask[2:22, 6:11, 6:11] = True
    spacing = np.array([0.002, 0.001, 0.0015])
    origin = np.array([-0.02, 0.3, -0.1])
    vertices, faces = mask_to_mesh(
        mask, spacing_zyx_mm=spacing * 1000, origin_xyz_mm=origin * 1000
    )
    grid = {"shape_zyx": mask.shape, "spacing_zyx_m": spacing, "origin_xyz_m": origin}
    recovered = topology.voxelize_mesh(vertices * 0.001, faces, **grid)
    np.testing.assert_array_equal(recovered, mask)
    result = topology.extract_centerlines(vertices * 0.001, faces, **grid)
    coordinates = np.argwhere(skeletonize(mask))
    np.testing.assert_allclose(
        result.points, origin + coordinates[:, ::-1] * spacing[::-1]
    )
    np.testing.assert_allclose(
        result.radii,
        distance_transform_edt(mask, sampling=spacing)[tuple(coordinates.T)],
    )
    assert len(result.edges) == len(result.points) - 1
    with pytest.raises(ValueError, match="Grid needs"):
        topology.voxelize_mesh(
            vertices * 0.001, faces, **{**grid, "spacing_zyx_m": [0, 1, 1]}
        )


def test_topology_selection_preserves_other_centerlines(monkeypatch):
    body = HumanBody(
        {
            n: AnatomicalStructure(
                n,
                Kind.VESSEL,
                np.ones((3, 3)),
                np.array([[0, 1, 2]]),
                centerline=graph(),
            )
            for n in ["aorta", "vascular_tree"]
        }
    )
    original = body.anatomy.structures["aorta"].centerline
    monkeypatch.setattr(topology, "extract_centerlines", lambda v, f, **grid: graph())
    assert set(body.extract_topology(names=["vascular_tree"])) == {"vascular_tree"}
    assert body.anatomy.structures["aorta"].centerline is original
    with pytest.raises(KeyError, match="Unknown anatomy"):
        body.extract_topology(names=["missing"])
