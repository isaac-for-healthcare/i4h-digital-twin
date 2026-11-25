# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-export quality check: repair simple mesh errors and bridge small vessel gaps.

``export_to_usd`` and ``export_patient_twin`` run ``repair`` (``HumanBody.repair``) first
unless called with ``repair_body=False``.

- ``repair_mesh`` drops faces on non-finite vertices, welds exactly coincident vertices
  (common in STL files), drops degenerate faces, repeated faces and back-to-back pairs
  (as ``mask_to_mesh`` does), drops disconnected islands under ``MIN_ISLAND_FRACTION`` of a
  structure's faces (stray segmentation voxels), and removes unreferenced vertices. A mesh
  without errors is left untouched.
- ``bridge_label_gaps`` joins vessel pieces that a segmentation gap separates, such as a
  subclavian artery labelled a voxel short of the aorta, before the navigation centerline is
  skeletonized. Only vessels of one circulation join (``vessels.circulation``): an artery never
  joins a vein, and the portal vein never joins the vena cava.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .body import HumanBody

MIN_ISLAND_FRACTION = 0.01
DEFAULT_MAX_GAP_M = 0.005


def _components(edges: NDArray[np.int64], count: int) -> NDArray[np.int32]:
    """Connected-component label of each of ``count`` nodes joined by ``edges``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(count, count))
    return connected_components(graph, directed=False)[1]


def repair_mesh(
    vertices: ArrayLike, faces: ArrayLike, *, min_island_fraction: float = MIN_ISLAND_FRACTION
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Fix simple triangle-mesh errors; returns ``(vertices, faces, fixes)``.

    ``fixes`` counts each repair (``non_finite_faces``, ``welded_vertices``,
    ``degenerate_faces``, ``back_to_back_faces``, ``duplicate_faces``, ``island_faces``,
    ``unreferenced_vertices``).
    When it is empty the inputs are returned unchanged. Dtypes are preserved.

    Raises:
        ValueError: If no valid face remains.
    """
    v, f = np.asarray(vertices), np.asarray(faces)
    if (v.ndim != 2 or v.shape[1:] != (3,) or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or f.dtype.kind not in "iu" or f.min() < 0 or f.max() >= len(v)):
        return v, f, {}  # Malformed meshes are left for export's validation to reject.
    out_v, out_f, fixes = v.astype(float), f.astype(np.int64).reshape(-1, 3), {}

    def drop(mask: NDArray[np.bool_], key: str) -> None:
        nonlocal out_f
        if mask.any():
            fixes[key] = int(mask.sum())
            out_f = out_f[~mask]

    drop((~np.isfinite(out_v).all(axis=1))[out_f].any(axis=1), "non_finite_faces")
    unique, inverse = np.unique(out_v, axis=0, return_inverse=True)
    if len(unique) < len(out_v):
        fixes["welded_vertices"] = len(out_v) - len(unique)
        out_v, out_f = unique, inverse.reshape(-1)[out_f]
    a, b, c = out_v[out_f[:, 0]], out_v[out_f[:, 1]], out_v[out_f[:, 2]]
    finite = out_v[np.isfinite(out_v).all(axis=1)]
    span = float((finite.max(axis=0) - finite.min(axis=0)).max()) if len(finite) else 0.0
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    repeated = (out_f[:, 0] == out_f[:, 1]) | (out_f[:, 1] == out_f[:, 2]) | (out_f[:, 0] == out_f[:, 2])
    drop(repeated | (area <= 1e-12 * span**2), "degenerate_faces")
    if not len(out_f):
        raise ValueError("Mesh has no valid faces")
    # Faces sharing a vertex triple: with both windings present they are a back-to-back pair that
    # encloses nothing (drop every copy, as mask_to_mesh does); otherwise keep the first copy.
    _, first, group = np.unique(np.sort(out_f, axis=1), axis=0, return_index=True, return_inverse=True)
    group = group.reshape(-1)
    shift = out_f.argmin(axis=1)  # A cyclic shift putting the smallest index first keeps the winding.
    ordered = np.take_along_axis(out_f, (shift[:, None] + np.arange(3)) % 3, axis=1)
    forward = np.bincount(group, weights=ordered[:, 1] < ordered[:, 2])
    back_to_back = ((forward > 0) & (forward < np.bincount(group)))[group]
    repeat = np.ones(len(out_f), bool)
    repeat[first] = False
    repeat &= ~back_to_back
    for mask, key in ((back_to_back, "back_to_back_faces"), (repeat, "duplicate_faces")):
        if mask.any():
            fixes[key] = int(mask.sum())
    out_f = out_f[~(back_to_back | repeat)]
    if not len(out_f):
        raise ValueError("Mesh has no valid faces")
    edges = np.concatenate([out_f[:, [0, 1]], out_f[:, [1, 2]]])
    face_component = _components(edges, len(out_v))[out_f[:, 0]]
    sizes = np.bincount(face_component)
    drop((sizes[face_component] < min_island_fraction * len(out_f)) & (face_component != np.argmax(sizes)), "island_faces")
    used = np.unique(out_f)
    if len(used) < len(out_v):
        fixes["unreferenced_vertices"] = len(out_v) - len(used)
        remap = np.full(len(out_v), -1, np.int64)
        remap[used] = np.arange(len(used))
        out_v, out_f = out_v[used], remap[out_f]
    if not fixes:
        return v, f, fixes
    return out_v.astype(v.dtype), out_f.astype(f.dtype), fixes


def _paint_tube(out: np.ndarray, a: NDArray, b: NDArray, radius: float, spacing: NDArray, label: int) -> None:
    """Set background voxels of ``out`` within ``radius`` of segment ``a``-``b`` (voxel indices) to ``label``."""
    reach = np.ceil(radius / spacing).astype(int)
    lo = np.maximum(np.minimum(a, b) - reach, 0)
    hi = np.minimum(np.maximum(a, b) + reach + 1, out.shape)
    box = tuple(slice(int(low), int(high)) for low, high in zip(lo, hi))
    grid = np.stack(np.meshgrid(*[np.arange(s.start, s.stop) for s in box], indexing="ij"), axis=-1) * spacing
    start, step = a * spacing, (b - a) * spacing
    t = np.clip((grid - start) @ step / max(float(step @ step), 1e-30), 0.0, 1.0)
    inside = np.linalg.norm(grid - (start + t[..., None] * step), axis=-1) <= radius
    region = out[box]
    region[inside & (region == 0)] = label


def bridge_label_gaps(
    labels: ArrayLike, spacing: ArrayLike, max_gap: float, *, circulation: ArrayLike | None = None
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Join vessel pieces separated by at most ``max_gap``, in a copy of a label volume.

    Args:
        labels: Integer volume; 0 is background.
        spacing: Voxel size along each array axis, in ``max_gap``'s units.
        max_gap: Largest distance between the nearest voxels of two pieces to bridge.
        circulation: Group of each label value (index 0 unused); pieces join only within a
            group. Default: one group.

    Within each group, the 26-connected pieces join the largest piece, closest first: a
    straight tube as wide as the narrower piece is painted between their nearest voxels with
    the joining piece's label, on background voxels only.

    Returns:
        ``(bridged labels, bridges)``; ``bridges`` lists ``(joining label, target label, gap)``.
    """
    from scipy import ndimage

    source = np.asarray(labels)
    source = source.astype(np.uint8) if source.dtype == bool else source
    sizes_mm = np.asarray(spacing, float)
    out = source.copy()
    bridges: list[tuple[int, int, float]] = []
    if not max_gap > 0 or not source.any():
        return out, bridges
    group_of = np.zeros(int(source.max()) + 1, np.int64) if circulation is None else np.asarray(circulation)
    groups = group_of[source]
    pad = np.ceil(max_gap / sizes_mm).astype(int) + 1
    for group in np.unique(group_of[np.unique(source[source > 0])]):
        mask = (source > 0) & (groups == group)
        piece, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
        if count < 2:
            continue
        sizes = np.bincount(piece.ravel())
        sizes[0] = 0
        main = int(np.argmax(sizes))
        joined, boxes, depth = piece == main, ndimage.find_objects(piece), None
        others = [c for c in range(1, count + 1) if c != main]
        while others:
            best = None
            for c in others:
                box = tuple(slice(max(s.start - p, 0), min(s.stop + p, n))
                            for s, p, n in zip(boxes[c - 1], pad, source.shape))
                target = joined[box]
                if not target.any():
                    continue
                distance, nearest = ndimage.distance_transform_edt(~target, sampling=sizes_mm, return_indices=True)
                distance = np.where(piece[box] == c, distance, np.inf)
                flat = int(np.argmin(distance))
                if distance.flat[flat] <= max_gap and (best is None or distance.flat[flat] < best[0]):
                    local = np.unravel_index(flat, distance.shape)
                    offset = np.array([s.start for s in box])
                    best = (float(distance.flat[flat]), c, np.array(local) + offset,
                            nearest[(slice(None), *local)] + offset, box)
            if best is None:
                break
            gap, c, a, b, box = best
            if depth is None:  # Distance to the background: each piece's inscribed radius.
                depth = ndimage.distance_transform_edt(mask, sampling=sizes_mm)
            radius = max(min(depth[piece == c].max(), depth[box][joined[box]].max()), sizes_mm.max())
            label = int(source[tuple(a)])
            _paint_tube(out, a, b, radius, sizes_mm, label)
            bridges.append((label, int(source[tuple(b)]), gap))
            joined |= piece == c
            others.remove(c)
    return out, bridges


def components(edges: ArrayLike, count: int) -> int:
    """Number of connected components among the ``count`` points that have edges."""
    e = np.asarray(edges, np.int64).reshape(-1, 2)
    if not len(e):
        return 0
    used = np.unique(e)
    return len(np.unique(_components(e, count)[used]))


def repair(body: HumanBody) -> dict[str, Any]:
    """Repair every structure mesh in place with ``repair_mesh``, hidden ones included.

    Returns:
        ``{"mesh_repairs": {name: fixes}}`` listing only changed structures.

    Example:
        ``body.repair()["mesh_repairs"]`` (same as ``quality.repair(body)``)
    """
    report: dict[str, Any] = {"mesh_repairs": {}}
    for name, structure in body.anatomy.structures.items():
        mesh = structure.mesh
        if mesh.vertices is not None and mesh.faces is not None:
            vertices, faces, fixes = repair_mesh(mesh.vertices, mesh.faces)
            if fixes:  # Assign on the MeshGeometry so a stored centerline is kept.
                mesh.vertices, mesh.faces = vertices, faces
                report["mesh_repairs"][name] = fixes
    return report
