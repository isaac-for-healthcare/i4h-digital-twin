# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry primitives shared by importers, HumanBody, and exporters.

Conventions: points are XYZ row vectors; voxel grids are indexed ZYX; 4x4
matrices map column vectors, so ``transform_points`` applies ``p @ M[:3, :3].T + M[:3, 3]``.

Main contents:

- ``CenterlineGraph``: a dataclass of centerline ``points`` (N, 3), per-point
  ``radii`` (N,), and undirected point-index ``edges`` (E, 2), in one frame and unit.
- ``transform_points`` / ``rigid_transform`` / ``validate_triangles``: apply,
  validate, and sanity-check affine transforms and triangle meshes.
- ``mask_to_mesh``: marching-cubes surface of a binary ZYX mask, in voxel-index XYZ.
- ``voxelize_mesh``: the inverse, rasterizing a closed mesh onto a ZYX grid (VTK).
- ``centerline_from_mask`` / ``extract_centerlines`` / ``native_centerline``:
  skeletonize a mask (or a voxelized mesh, or a native scan-grid mask) into a
  26-connected graph whose radii come from the Euclidean distance transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .scan_volume import ScanVolume


@dataclass
class CenterlineGraph:
    """Centerline points (N, 3) and radii (N,) in one frame/unit, plus (E, 2) point-index edges.

    Graphs stored on an ``AnatomicalStructure`` use that structure's local XYZ meters.
    """

    points: np.ndarray
    radii: np.ndarray
    edges: np.ndarray


def transform_points(points: ArrayLike, matrix: NDArray[np.floating]) -> NDArray[np.floating]:
    """Apply a 4x4 affine to (N, 3) XYZ points; output units follow the matrix.

    Example:
        ``world = transform_points(structure.vertices, structure.local_to_world)``
    """
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def rigid_transform(value: ArrayLike) -> NDArray[np.float64]:
    """Return a validated float copy of a proper rigid 4x4 transform.

    Args:
        value: Any 4x4 array-like. Rotation must be orthonormal with determinant +1.

    Raises:
        ValueError: For non-finite values, scale, shear, reflection, or a bad last row.
    """
    m = np.array(value, dtype=float, copy=True)
    if (m.shape != (4, 4) or not np.isfinite(m).all()
            or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(m[:3, :3]), 1, atol=2e-5)):
        raise ValueError("Expected a finite proper rigid 4x4 transform (no scale/reflection)")
    return m


def validate_triangles(
    vertices: ArrayLike, faces: ArrayLike, *, name: str = "mesh"
) -> tuple[NDArray[np.float64], NDArray[np.integer]]:
    """Return ``(float vertices (N, 3), integer faces (F, 3))`` or raise ``ValueError``.

    ``name`` only labels the error message. Requires at least three finite vertices,
    at least one face, and face indices within range.
    """
    v, f = np.asarray(vertices, dtype=float), np.asarray(faces)
    if (v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 3 or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
            or not np.issubdtype(f.dtype, np.integer) or f.min() < 0 or f.max() >= len(v)):
        raise ValueError(f"Invalid triangle mesh: {name}")
    return v, f


def mask_to_mesh(mask_zyx: ArrayLike) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Extract a closed, outward-wound triangle surface from a binary ZYX mask.

    Args:
        mask_zyx: 3D array of 0/1 (or bool) indexed ``[z, y, x]`` with at least one voxel set.

    Returns:
        ``(vertices, faces)``: float32 (N, 3) vertices in voxel-index XYZ (voxel centers
        are integers) and int32 (F, 3) triangles. The mask is padded so structures that
        touch the boundary stay closed. Apply the image's voxel-to-world affine afterwards.

    Example:
        ``vertices, faces = mask_to_mesh(labels == 5)``
    """
    from skimage.measure import marching_cubes

    mask = np.asarray(mask_zyx)
    if mask.ndim != 3 or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError(f"Expected a nonempty binary 3D mask, got shape {mask.shape}")
    vertices, faces, _, _ = marching_cubes(np.pad(mask.astype(np.uint8), 1), level=0.5, allow_degenerate=False)
    return (vertices - 1)[:, ::-1].astype(np.float32), faces.astype(np.int32)


def voxelize_mesh(
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    shape_zyx: tuple[int, ...],
    spacing_zyx_m: ArrayLike,
    origin_xyz_m: ArrayLike,
) -> NDArray[np.bool_]:
    """Rasterize a closed manifold triangle mesh onto an axis-aligned ZYX grid (requires VTK).

    Args:
        vertices, faces: The mesh, in the same frame/units as the grid.
        shape_zyx: Grid size; spacing_zyx_m: voxel size (ZYX); origin_xyz_m: the XYZ
            position of voxel (0, 0, 0)'s center.

    Returns:
        Boolean mask of shape ``shape_zyx``.

    Raises:
        ValueError: For degenerate triangles or an open / non-manifold surface.
    """
    try:
        import vtk
        from vtk.util.numpy_support import (
            numpy_to_vtk,
            numpy_to_vtkIdTypeArray,
            vtk_to_numpy,
        )
    except ImportError as exc:
        raise ImportError("Mesh skeletonization requires VTK: pip install vtk") from exc
    shape, spacing = np.asarray(shape_zyx), np.asarray(spacing_zyx_m, dtype=float)
    v, f = validate_triangles(vertices, faces)
    t = v[f]
    if np.any(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) == 0):
        raise ValueError("Mesh contains degenerate triangles")
    points, cells, surface = vtk.vtkPoints(), vtk.vtkCellArray(), vtk.vtkPolyData()
    points.SetData(numpy_to_vtk(v, deep=True))
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 3 * (len(f) + 1), 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(f.astype(np.int64).ravel(), deep=True),
    )
    surface.SetPoints(points)
    surface.SetPolys(cells)
    clean = vtk.vtkCleanPolyData()  # Welds coincident STL vertices.
    clean.SetInputData(surface)
    clean.Update()
    boundary = vtk.vtkFeatureEdges()
    boundary.SetInputData(clean.GetOutput())
    boundary.BoundaryEdgesOn()
    boundary.NonManifoldEdgesOn()
    boundary.FeatureEdgesOff()
    boundary.ManifoldEdgesOff()
    boundary.Update()
    if boundary.GetOutput().GetNumberOfCells():
        raise ValueError("Skeleton voxelization requires a closed manifold surface")
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(clean.GetOutput())
    stencil.SetOutputOrigin(*np.asarray(origin_xyz_m, dtype=float))
    stencil.SetOutputSpacing(*spacing[::-1])
    stencil.SetOutputWholeExtent(0, int(shape[2]) - 1, 0, int(shape[1]) - 1, 0, int(shape[0]) - 1)
    image = vtk.vtkImageStencilToImage()
    image.SetInputConnection(stencil.GetOutputPort())
    image.SetInsideValue(1)
    image.SetOutsideValue(0)
    image.SetOutputScalarTypeToUnsignedChar()
    image.Update()
    return vtk_to_numpy(image.GetOutput().GetPointData().GetScalars()).reshape(tuple(shape)).astype(bool)


def centerline_from_mask(
    mask_zyx: ArrayLike, spacing_zyx: ArrayLike, origin_xyz: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Skeletonize a binary ZYX mask into a centerline graph.

    Args:
        mask_zyx: Binary 3D mask. It is never closed, resampled, or reduced to one component.
        spacing_zyx: Voxel size (ZYX); origin_xyz: XYZ position of voxel (0, 0, 0)'s center.

    Returns:
        ``(points, edges, radii)``: XYZ skeleton voxel centers, 26-connected index pairs
        ``(i, j)`` with ``i < j``, and distance-transform radii, all in the spacing's units.

    Raises:
        ValueError: If the skeleton has fewer than two voxels or no edges.
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    mask = np.asarray(mask_zyx)
    spacing, origin = np.asarray(spacing_zyx, float), np.asarray(origin_xyz, float)
    coords = np.argwhere(skeletonize(mask.astype(bool)))
    if len(coords) < 2:
        raise ValueError(f"Skeleton has too few nodes ({len(coords)}).")
    edt = ndimage.distance_transform_edt(mask.astype(bool), sampling=tuple(spacing))
    points = origin + coords[:, ::-1] * spacing[::-1]
    index = {tuple(c): i for i, c in enumerate(coords.tolist())}
    offsets = [(z, y, x) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1) if z or y or x]
    edges = [
        (i, j)
        for i, (z, y, x) in enumerate(coords.tolist())
        for dz, dy, dx in offsets
        if (j := index.get((z + dz, y + dy, x + dx), -1)) > i
    ]
    if not edges:
        raise ValueError("Skeleton produced no edges.")
    return points, np.asarray(edges, dtype=np.int64), edt[tuple(coords.T)]


def extract_centerlines(
    vertices: ArrayLike,
    faces: ArrayLike,
    *,
    shape_zyx: tuple[int, ...],
    spacing_zyx_m: ArrayLike,
    origin_xyz_m: ArrayLike,
) -> CenterlineGraph:
    """Centerline of a closed tubular mesh: ``voxelize_mesh`` then ``centerline_from_mask``.

    Usually called through ``HumanBody.extract_topology``, which builds a bounded grid
    around each mesh. Grid arguments are as for ``voxelize_mesh``.
    """
    mask = voxelize_mesh(vertices, faces, shape_zyx=shape_zyx,
                         spacing_zyx_m=spacing_zyx_m, origin_xyz_m=origin_xyz_m)
    points, edges, radii = centerline_from_mask(mask, spacing_zyx_m, origin_xyz_m)
    return CenterlineGraph(points=points, radii=radii, edges=edges)


def native_centerline(
    mask: NDArray, scan: ScanVolume
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float32]]:
    """Centerline of a binary mask on a scan's native grid, in the scan's world frame and units.

    Args:
        mask: Binary array with the same shape and array axis order as ``scan.values``.
        scan: The ``ScanVolume`` whose grid the mask lives on.

    Returns:
        ``(points, edges, radii)`` as written to a bundle's ``centerline_*.npy`` files.

    Raises:
        ValueError: If the scan's voxel axes are not orthogonal (sheared grids).
    """
    a = scan.ijk_to_world
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    direction = a[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Centerline radii require orthogonal voxel axes; reconstruct sheared grids")
    # Thinning depends on voxel traversal order. Orient a temporary view toward
    # LPS for reproducible graphs, without resampling or changing exported arrays.
    from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation

    ijk = mask.transpose([scan.array_axes.index(c) for c in "ijk"])
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ scan.ijk_to_ras_m
    orientation = io_orientation(lps)
    ordered = apply_orientation(ijk, orientation)
    ordered_to_scan = a @ inv_ornt_aff(orientation, ijk.shape)
    ordered_spacing = np.linalg.norm(ordered_to_scan[:3, :3], axis=0)
    points, edges, radii = centerline_from_mask(
        ordered.transpose(2, 1, 0), ordered_spacing[::-1], (0.0, 0.0, 0.0)
    )
    points = points @ (ordered_to_scan[:3, :3] / ordered_spacing).T + ordered_to_scan[:3, 3]
    return points, edges, radii.astype(np.float32)
