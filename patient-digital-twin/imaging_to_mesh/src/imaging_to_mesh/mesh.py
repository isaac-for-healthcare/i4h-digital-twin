# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Surface extraction and OBJ export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def mask_to_mesh(
    mask_zyx: np.ndarray,
    *,
    spacing_zyx_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an XYZ surface mesh from a 3D binary ZYX mask.

    Args:
        mask_zyx: Binary volume indexed as ``(z, y, x)``.
        spacing_zyx_mm: Voxel spacing in millimeters.
        origin_xyz_mm: World-space origin in millimeters.

    Returns:
        ``(vertices_xyz_mm, triangle_indices)``.
    """
    mask = np.asarray(mask_zyx, dtype=np.uint8)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")
    if not np.any(mask):
        raise ValueError("Cannot extract a mesh from an empty mask.")

    # Padding closes structures that touch an image boundary.
    padded = np.pad(mask, 1)
    vertices_zyx, faces, _, _ = marching_cubes(
        padded,
        level=0.5,
        spacing=spacing_zyx_mm,
        allow_degenerate=False,
    )
    vertices_zyx -= np.asarray(spacing_zyx_mm, dtype=np.float32)
    vertices_xyz = vertices_zyx[:, ::-1]
    vertices_xyz += np.asarray(origin_xyz_mm, dtype=np.float32)
    return vertices_xyz.astype(np.float32), faces.astype(np.int32)


def write_obj(vertices_xyz: np.ndarray, faces: np.ndarray, output_path: str | Path) -> Path:
    """Write one triangle mesh as Wavefront OBJ."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices_xyz, faces=faces, process=False)
    mesh.export(output, file_type="obj")
    return output
