# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tubular centerlines from skeletonized voxel masks.

Meshes are voxelized on an explicit grid with VTK, then thinned; native scan
masks are thinned directly. Radii come from the Euclidean distance transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import validate_triangles


@dataclass
class CenterlineGraph:
    """Local XYZ-meter points, meter radii, and undirected point-index edges."""

    points: np.ndarray
    radii: np.ndarray
    edges: np.ndarray


def _surface(vertices, faces, vtk):
    """Validate and copy triangular geometry; weld coincident STL vertices."""
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    vertices, faces = validate_triangles(vertices, faces)
    triangles = vertices[faces]
    if np.any(
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        == 0
    ):
        raise ValueError("Mesh contains degenerate triangles")
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(
            np.arange(0, 3 * (len(faces) + 1), 3, dtype=np.int64), deep=True
        ),
        numpy_to_vtkIdTypeArray(faces.astype(np.int64).ravel(), deep=True),
    )
    surface = vtk.vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(cells)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(surface)
    clean.Update()
    return clean.GetOutput()


def voxelize_mesh(vertices, faces, *, shape_zyx, spacing_zyx_m, origin_xyz_m):
    """Sample a closed mesh on an explicit axis-aligned local grid using VTK.

    The grid origin is the center of voxel (0, 0, 0), not a voxel corner.
    No smoothing, closing, or largest-component filtering is applied here.
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise ImportError("Mesh skeletonization requires VTK: pip install vtk") from exc
    shape = np.asarray(shape_zyx)
    spacing = np.asarray(spacing_zyx_m, dtype=float)
    origin = np.asarray(origin_xyz_m, dtype=float)
    if (
        shape.shape != (3,)
        or not np.issubdtype(shape.dtype, np.integer)
        or np.any(shape <= 0)
        or spacing.shape != (3,)
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0)
        or origin.shape != (3,)
        or not np.isfinite(origin).all()
    ):
        raise ValueError(
            "Grid needs positive integer ZYX shape, positive meter spacing and finite XYZ origin"
        )
    surface = _surface(vertices, faces, vtk)
    boundary = vtk.vtkFeatureEdges()
    boundary.SetInputData(surface)
    boundary.BoundaryEdgesOn()
    boundary.NonManifoldEdgesOn()
    boundary.FeatureEdgesOff()
    boundary.ManifoldEdgesOff()
    boundary.Update()
    if boundary.GetOutput().GetNumberOfCells():
        raise ValueError("Skeleton voxelization requires a closed manifold surface")
    extent = (0, int(shape[2]) - 1, 0, int(shape[1]) - 1, 0, int(shape[0]) - 1)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(surface)
    stencil.SetOutputOrigin(*origin)
    stencil.SetOutputSpacing(*spacing[::-1])
    stencil.SetOutputWholeExtent(*extent)
    stencil.Update()
    image = vtk.vtkImageStencilToImage()
    image.SetInputConnection(stencil.GetOutputPort())
    image.SetInsideValue(1)
    image.SetOutsideValue(0)
    image.SetOutputScalarTypeToUnsignedChar()
    image.Update()
    return (
        vtk_to_numpy(image.GetOutput().GetPointData().GetScalars())
        .reshape(tuple(shape))
        .astype(bool)
    )


def extract_centerlines(vertices, faces, *, shape_zyx, spacing_zyx_m, origin_xyz_m):
    """Voxelize a closed local-meter mesh on the given grid and skeletonize it."""
    mask = voxelize_mesh(vertices, faces, shape_zyx=shape_zyx,
                         spacing_zyx_m=spacing_zyx_m, origin_xyz_m=origin_xyz_m)
    points, edges, radii = centerline_from_mask(mask, spacing_zyx_m, origin_xyz_m)
    return CenterlineGraph(points=points, radii=radii, edges=edges)


def centerline_from_mask(
    mask_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Skeleton points, edges, and EDT radii in the spacing/origin's spatial units.

    Integer indices locate voxel centers. The mask is never closed, resampled,
    reduced to its largest component, or subsampled.
    """
    mask_zyx = np.asarray(mask_zyx)
    spacing = np.asarray(spacing_zyx, float)
    origin = np.asarray(origin_xyz, float)
    if (mask_zyx.ndim != 3 or not mask_zyx.size or not np.isin(mask_zyx, [0, 1]).all()
            or not mask_zyx.any()):
        raise ValueError("Expected a nonempty binary 3D mask")
    if (spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or origin.shape != (3,) or not np.isfinite(origin).all()):
        raise ValueError("Expected positive finite spacing and a finite XYZ origin")
    from scipy import ndimage
    from skimage.morphology import skeletonize

    skel = skeletonize(mask_zyx.astype(bool))

    sz, sy, sx = (float(v) for v in spacing_zyx)
    ox, oy, oz = (float(v) for v in origin_xyz)

    edt = ndimage.distance_transform_edt(mask_zyx.astype(bool), sampling=(sz, sy, sx))
    coords = np.argwhere(skel)
    if coords.shape[0] < 2:
        raise ValueError(f"Skeleton has too few nodes ({coords.shape[0]}).")

    index_of = {(int(z), int(y), int(x)): i for i, (z, y, x) in enumerate(coords)}
    pts = np.empty((coords.shape[0], 3), dtype=np.float64)
    pts[:, 0] = ox + coords[:, 2] * sx
    pts[:, 1] = oy + coords[:, 1] * sy
    pts[:, 2] = oz + coords[:, 0] * sz
    radii = edt[coords[:, 0], coords[:, 1], coords[:, 2]]

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
        raise ValueError("Skeleton produced no edges.")
    return pts, edges_arr, radii


def native_centerline(mask, scan):
    """Extract on the native mask; map voxel centres and radii into scan units."""
    a = scan.ijk_to_world
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    direction = a[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError(
            "Centerline radii require orthogonal voxel axes; explicitly reconstruct sheared grids"
        )
    kji = mask.transpose([scan.array_axes.index(c) for c in "kji"])
    points, edges, radii = centerline_from_mask(kji, spacing[::-1], (0.0, 0.0, 0.0))
    points = points.astype(np.float32).astype(float) @ direction.T + a[:3, 3]
    return points, edges, radii.astype(np.float32)
