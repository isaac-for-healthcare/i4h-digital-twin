# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Affine math, mesh validation, mask surfaces, voxelization, and skeleton centerlines."""

import numpy as np
import pytest
from patient_digital_twin.geometry import (
    extract_centerlines,
    mask_to_mesh,
    rigid_transform,
    transform_points,
    validate_triangles,
    voxelize_mesh,
)


def test_rigid_inverse_roundtrip_and_copy():
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    matrix[:3, 3] = [3, 4, 5]
    target = transform_points(source, matrix)
    np.testing.assert_allclose(transform_points(target, np.linalg.inv(matrix)), source)
    copy = rigid_transform(matrix)
    copy[0, 3] = 20
    assert matrix[0, 3] == 3


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3),
        np.diag([2.0, 1.0, 1.0, 1.0]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
        np.diag([1.0, 1.0, 1.0, 0.0]),
        np.full((4, 4), np.nan),
        np.array([[1.0, 0.2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    ],
)
def test_invalid_rigid_frames_rejected(matrix):
    with pytest.raises(ValueError, match="rigid"):
        rigid_transform(matrix)


def test_triangle_validation():
    vertices, faces = validate_triangles([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    assert vertices.dtype == float and faces.tolist() == [[0, 1, 2]]


@pytest.mark.parametrize(
    "vertices, faces",
    [
        (np.zeros((2, 3)), [[0, 1, 1]]),
        (np.full((3, 3), np.nan), [[0, 1, 2]]),
        (np.zeros((3, 3)), [[0, 1, 3]]),
        (np.zeros((3, 3)), [[0.0, 1.0, 2.0]]),
        (np.zeros((3, 3)), np.zeros((0, 3), int)),
    ],
)
def test_invalid_triangles_rejected(vertices, faces):
    with pytest.raises(ValueError, match="Invalid triangle mesh"):
        validate_triangles(vertices, faces)


def test_boundary_mesh_is_closed_outward_in_voxel_xyz():
    mask = np.ones((2, 3, 4), dtype=bool)
    vertices, faces = mask_to_mesh(mask)
    assert vertices.dtype == np.float32 and faces.dtype == np.int32
    np.testing.assert_allclose(vertices.min(0), [-0.5, -0.5, -0.5])
    np.testing.assert_allclose(vertices.max(0), [3.5, 2.5, 1.5])
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert (counts == 2).all()
    triangles = vertices[faces] - vertices.mean(0)
    assert np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() > 0
    assert mask.all()  # Extraction must not mutate its input.


@pytest.mark.parametrize(
    "mask",
    [np.zeros((3, 3, 3)), np.ones((3, 3)), np.empty((0, 2, 3)), np.full((3, 3, 3), 2),
     np.full((3, 3, 3), 0.5), np.full((3, 3, 3), np.nan), np.full((3, 3, 3), -1)],
)
def test_invalid_masks_rejected(mask):
    with pytest.raises(ValueError):
        mask_to_mesh(mask)


def test_mesh_skeleton_grid_round_trip_and_physical_radii():
    pytest.importorskip("vtk")
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    mask = np.zeros((24, 17, 17), dtype=bool)
    mask[2:22, 6:11, 6:11] = True
    spacing, origin = np.array([0.002, 0.001, 0.0015]), np.array([-0.02, 0.3, -0.1])
    vertices, faces = mask_to_mesh(mask)
    vertices = origin + vertices * spacing[::-1]
    grid = {"shape_zyx": mask.shape, "spacing_zyx_m": spacing, "origin_xyz_m": origin}
    np.testing.assert_array_equal(voxelize_mesh(vertices, faces, **grid), mask)
    result = extract_centerlines(vertices, faces, **grid)
    coordinates = np.argwhere(skeletonize(mask))
    np.testing.assert_allclose(result.points, origin + coordinates[:, ::-1] * spacing[::-1])
    np.testing.assert_allclose(result.radii, distance_transform_edt(mask, sampling=spacing)[tuple(coordinates.T)])
    assert len(result.edges) == len(result.points) - 1
    open_faces = faces[1:]
    with pytest.raises(ValueError, match="closed manifold"):
        voxelize_mesh(vertices, open_faces, **grid)
