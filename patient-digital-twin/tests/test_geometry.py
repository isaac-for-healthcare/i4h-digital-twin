# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy-only tests for affine math and mesh validation."""

import numpy as np
import pytest
from patient_digital_twin.geometry import (
    rigid_transform,
    transform_points,
    validate_triangles,
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
