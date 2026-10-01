# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voxel skeleton and graph artifacts in physical patient millimeters."""

import numpy as np


def centerline_from_mask(
    mask_zyx: np.ndarray,
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    max_nodes: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy import ndimage

    try:
        from skimage.morphology import skeletonize

        skel = skeletonize(mask_zyx.astype(bool))
    except TypeError:
        from skimage.morphology import skeletonize_3d

        skel = skeletonize_3d(mask_zyx.astype(bool)) > 0

    sz, sy, sx = (float(v) for v in spacing_zyx_mm)
    ox, oy, oz = (float(v) for v in origin_xyz_mm)

    edt = ndimage.distance_transform_edt(mask_zyx.astype(bool), sampling=(sz, sy, sx))
    coords = np.argwhere(skel)
    if coords.shape[0] < 4:
        raise RuntimeError(f"Skeleton has too few nodes ({coords.shape[0]}).")

    if max_nodes and coords.shape[0] > max_nodes:
        sel = np.linspace(0, coords.shape[0] - 1, max_nodes).round().astype(np.int64)
        sel = np.unique(sel)
        coords = coords[sel]

    index_of = {(int(z), int(y), int(x)): i for i, (z, y, x) in enumerate(coords)}
    pts = np.empty((coords.shape[0], 3), dtype=np.float32)
    pts[:, 0] = ox + coords[:, 2] * sx
    pts[:, 1] = oy + coords[:, 1] * sy
    pts[:, 2] = oz + coords[:, 0] * sz
    radii = edt[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)

    neighbors = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]
    edges: list[tuple[int, int]] = []
    for i, (z, y, x) in enumerate(coords):
        for dz, dy, dx in neighbors:
            j = index_of.get((int(z + dz), int(y + dy), int(x + dx)))
            if j is not None and j > i:
                edges.append((i, j))
    edges_arr = (
        np.asarray(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
    )
    if edges_arr.shape[0] < 1:
        raise RuntimeError("Skeleton produced no edges.")
    return pts, edges_arr, radii
