# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal surface extraction for patient segmentation masks; no file exporters."""

from __future__ import annotations

import numpy as np
from skimage.measure import marching_cubes


def mask_to_mesh(
    mask_zyx: np.ndarray,
    *,
    spacing_zyx_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an outward XYZ triangle surface from a binary ZYX volume.

    Padding closes structures touching the scan boundary. Spacing is ordered
    ZYX in millimeters; origin is XYZ in millimeters. Return float32 vertices
    and int32 triangle indices. No files are written and no USD runtime is used.

    SegmentationImporter uses unit spacing/zero origin to get voxel XYZ, then
    applies the full imaging affine itself (including orientation and shear).
    For direct physical output, supply positive spacing and a finite origin.
    """
    mask = np.asarray(mask_zyx)
    if mask.ndim != 3 or min(mask.shape) == 0:
        raise ValueError(f"Expected a nonempty 3D mask, got shape {mask.shape}")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected a finite binary mask containing only 0 and 1")
    if not np.any(mask):
        raise ValueError("Cannot extract a mesh from an empty mask.")
    spacing = np.asarray(spacing_zyx_mm, dtype=float)
    origin = np.asarray(origin_xyz_mm, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Voxel spacing must be three finite positive values")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Origin must be three finite XYZ values")

    # Keep the original marching-cubes/padding contract so existing fitted
    # patient meshes, topology, and rigid anchor frames remain reproducible.
    padded = np.pad(mask.astype(np.uint8), 1)
    vertices_zyx, faces, _, _ = marching_cubes(
        padded,
        level=0.5,
        spacing=spacing,
        allow_degenerate=False,
    )
    vertices_zyx -= spacing.astype(np.float32)
    vertices_xyz = vertices_zyx[:, ::-1]
    vertices_xyz += origin.astype(np.float32)
    return vertices_xyz.astype(np.float32), faces.astype(np.int32)
