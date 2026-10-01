# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate transforms and landmark registration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def to_numpy(value) -> np.ndarray:
    """Detach a tensor to CPU NumPy or coerce array-like input; copy if mutating."""
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


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


@dataclass(frozen=True)
class Similarity:
    """Uniform-scale landmark fit with rotation and translation."""

    rotation: np.ndarray
    scale: float
    offset: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map source XYZ points into the fitted target frame, including uniform scale."""
        return np.asarray(points) @ (self.scale * self.rotation).T + self.offset

    def as_4x4(self) -> np.ndarray:
        """Return the homogeneous affine; it is rigid only when scale equals one."""
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = self.scale * self.rotation, self.offset
        return matrix


def solve_similarity(source, target, *, fit_scale: bool = True) -> Similarity:
    """Fit corresponding (N, 3) XYZ landmarks, rejecting degenerate inputs.

    Use fit_scale=True to estimate uniform size; use False for an imaging-to-
    model registration that preserves the physical dimensions of source meshes.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or source.shape != target.shape
        or len(source) < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Need at least 3 finite corresponding XYZ landmarks")
    src, dst = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(src) < 2 or np.linalg.matrix_rank(dst) < 2:
        raise ValueError("Landmarks must be non-collinear")
    left, values, right = np.linalg.svd(dst.T @ src / len(src))
    correction = np.eye(3)
    correction[2, 2] = 1 if np.linalg.det(left @ right) > 0 else -1
    rotation = left @ correction @ right
    scale = (
        float(np.sum(values * np.diag(correction)) / np.mean(np.sum(src**2, axis=1)))
        if fit_scale
        else 1.0
    )
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Invalid landmark scale")
    return Similarity(
        rotation, scale, target.mean(0) - scale * rotation @ source.mean(0)
    )


def rotation_between(source, target) -> np.ndarray:
    """Return a rotation vector in radians aligning two nonzero XYZ directions."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if min(np.linalg.norm(source), np.linalg.norm(target)) < 1e-12:
        raise ValueError("Cannot rotate a zero direction")
    source, target = source / np.linalg.norm(source), target / np.linalg.norm(target)
    cross, cosine = np.cross(source, target), float(np.clip(source @ target, -1, 1))
    sine = np.linalg.norm(cross)
    if sine < 1e-9:
        if cosine > 0:
            return np.zeros(3)
        cross = np.cross(source, np.eye(3)[np.argmin(np.abs(source))])
        return cross / np.linalg.norm(cross) * np.pi
    return cross / sine * np.arctan2(sine, cosine)
