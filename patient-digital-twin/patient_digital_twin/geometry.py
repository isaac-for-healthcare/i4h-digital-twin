# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate transforms and triangle-mesh validation."""

from __future__ import annotations

import numpy as np


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine to XYZ points using row vectors; units follow the matrix."""
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def rigid_transform(value) -> np.ndarray:
    """Validate and copy a proper 4x4 rigid frame; reject scale, shear and reflection."""
    matrix = np.array(value, dtype=float, copy=True)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-5)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=2e-5)
    ):
        raise ValueError(
            "Expected a finite proper rigid 4x4 transform (no scale/reflection)"
        )
    return matrix


def validate_triangles(vertices, faces, *, name="mesh", min_vertices=3):
    """Return float vertices and integer faces, or raise for an invalid triangle mesh."""
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) < min_vertices
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError(f"Invalid triangle mesh: {name}")
    return vertices, faces
