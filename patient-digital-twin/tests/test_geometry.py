# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy-only tests for affine math and similarity value objects."""

import numpy as np
import pytest
from patient_digital_twin import Similarity
from patient_digital_twin.geometry import (
    rigid_transform,
    rotation_between,
    solve_similarity,
    to_numpy,
    transform_points,
)


def test_similarity_apply_matches_homogeneous_affine():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    fit = Similarity(rotation, 2.0, np.array([3.0, 4.0, 5.0]))
    points = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(
        fit.apply(points), transform_points(points, fit.as_4x4())
    )
    np.testing.assert_allclose(fit.as_4x4()[3], [0, 0, 0, 1])
    with pytest.raises(ValueError, match="rigid"):
        rigid_transform(fit.as_4x4())


def test_rigid_fit_and_inverse_roundtrip():
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    matrix[:3, 3] = [3, 4, 5]
    target = transform_points(source, matrix)
    fit = solve_similarity(source, target, fit_scale=False)
    assert fit.scale == 1
    np.testing.assert_allclose(fit.as_4x4(), matrix, atol=1e-12)
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


def test_direction_rotations_and_numpy_coercion():
    np.testing.assert_allclose(
        rotation_between([1, 0, 0], [0, 1, 0]), [0, 0, np.pi / 2]
    )
    np.testing.assert_allclose(rotation_between([1, 0, 0], [2, 0, 0]), [0, 0, 0])
    with pytest.raises(ValueError, match="zero"):
        rotation_between([0, 0, 0], [1, 0, 0])
    np.testing.assert_array_equal(to_numpy([[1, 2, 3]]), [[1, 2, 3]])


@pytest.mark.parametrize(
    "source",
    [
        np.zeros((2, 3)),
        np.zeros((3, 2)),
        np.full((3, 3), np.nan),
    ],
)
def test_similarity_rejects_invalid_correspondences(source):
    with pytest.raises(ValueError, match="landmarks"):
        solve_similarity(source, np.zeros((3, 3)))
