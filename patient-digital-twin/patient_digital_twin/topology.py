# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic tubular mesh centerlines, following the vasculature VMTK workflow.

VMTK's automatic network extractor supplies seeds for its centerline algorithm;
no interactive seed selection is used. VTK/VMTK are optional runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CenterlineGraph:
    """Local XYZ-meter points, meter radii, and undirected point-index edges."""

    points: np.ndarray
    radii: np.ndarray
    edges: np.ndarray


def _runtime():
    try:
        import vtk
        from vmtk import vmtkscripts
    except ImportError as exc:
        raise ImportError(
            "Centerline extraction requires optional VTK and VMTK. Install a "
            "compatible VMTK distribution in this Python environment: "
            "https://www.vmtk.org/download/"
        ) from exc
    return vtk, vmtkscripts


def _surface(vertices, faces, vtk):
    """Validate and copy triangular geometry; weld coincident STL vertices."""
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or len(vertices) < 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError(
            "Expected finite XYZ vertices and valid integer triangle indices"
        )
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


def _graph(polydata):
    """Convert VMTK polylines without inventing radii when extraction fails."""
    from vtk.util.numpy_support import vtk_to_numpy

    if polydata is None or polydata.GetNumberOfPoints() < 2:
        raise ValueError("VMTK did not produce a centerline")
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float, copy=True)
    radius = polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if radius is None:
        raise ValueError("VMTK output has no MaximumInscribedSphereRadius array")
    radii = vtk_to_numpy(radius).astype(float, copy=True).reshape(-1)
    edges = set()
    for index in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(index)
        if cell.GetCellDimension() != 1:
            continue
        for j in range(cell.GetNumberOfPoints() - 1):
            a, b = cell.GetPointId(j), cell.GetPointId(j + 1)
            if a != b:
                edges.add(tuple(sorted((a, b))))
    if (
        len(radii) != len(points)
        or not np.isfinite(points).all()
        or not np.isfinite(radii).all()
        or np.any(radii < 0)
        or not edges
    ):
        raise ValueError("VMTK produced an invalid or empty centerline graph")
    return CenterlineGraph(points, radii, np.asarray(sorted(edges), dtype=np.int64))


def extract_centerlines(vertices, faces, *, method="vmtk", **grid) -> CenterlineGraph:
    """Extract every connected tubular component from a mesh in local meters.

    Handles capped and open surfaces using automatic VMTK seed discovery. Works
    on copies: upstream network extraction may open a hole in its input surface.
    Invalid/non-tubular surfaces may fail; no centerline is fabricated for them.
    Alternatively, method="skeleton" samples a closed mesh on the caller's
    shape_zyx, spacing_zyx_m, origin_xyz_m grid and thins that mask, without VMTK.
    """
    if method == "skeleton":
        return _skeleton_centerlines(vertices, faces, **grid)
    if method != "vmtk" or grid:
        raise ValueError(
            "Use method='vmtk' without grid options or method='skeleton' with an explicit grid"
        )
    vtk, scripts = _runtime()
    surface = _surface(vertices, faces, vtk)
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(surface)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    graphs = []
    for region in range(connectivity.GetNumberOfExtractedRegions()):
        component = vtk.vtkPolyDataConnectivityFilter()
        component.SetInputData(surface)
        component.SetExtractionModeToSpecifiedRegions()
        component.AddSpecifiedRegion(region)
        component.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(component.GetOutputPort())
        clean.Update()
        copied = vtk.vtkPolyData()
        copied.DeepCopy(clean.GetOutput())
        extractor = scripts.vmtkCenterlinesNetwork()
        extractor.Surface = copied
        extractor.UseJoblib = False
        extractor.RandomSeed = 0
        extractor.LogOn = 0
        extractor.Execute()
        graphs.append(_graph(extractor.Centerlines))
    if not graphs:
        raise ValueError("Mesh contains no connected surface")
    offsets = np.cumsum([0] + [len(graph.points) for graph in graphs[:-1]])
    return CenterlineGraph(
        np.concatenate([graph.points for graph in graphs]),
        np.concatenate([graph.radii for graph in graphs]),
        np.concatenate(
            [graph.edges + offset for graph, offset in zip(graphs, offsets)]
        ),
    )


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


def _skeleton_centerlines(vertices, faces, *, shape_zyx, spacing_zyx_m, origin_xyz_m):
    mask = voxelize_mesh(vertices, faces, shape_zyx=shape_zyx,
                         spacing_zyx_m=spacing_zyx_m, origin_xyz_m=origin_xyz_m)
    points, edges, radii = centerline_from_mask(mask, spacing_zyx_m, origin_xyz_m)
    return CenterlineGraph(points, radii, edges)


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

    try:
        from skimage.morphology import skeletonize

        skel = skeletonize(mask_zyx.astype(bool))
    except TypeError:
        from skimage.morphology import skeletonize_3d

        skel = skeletonize_3d(mask_zyx.astype(bool)) > 0

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
