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
    """Voxel thinning and EDT radii, matching i4h-workflows' 26-neighbor graph."""
    from itertools import product

    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    mask = voxelize_mesh(
        vertices,
        faces,
        shape_zyx=shape_zyx,
        spacing_zyx_m=spacing_zyx_m,
        origin_xyz_m=origin_xyz_m,
    )
    coordinates = np.argwhere(skeletonize(mask))
    if len(coordinates) < 2:
        raise ValueError("Mesh skeleton has fewer than two points")
    distance = distance_transform_edt(mask, sampling=spacing_zyx_m)
    index = {tuple(c): i for i, c in enumerate(coordinates)}
    edges = []
    offsets = [np.array(o) for o in product((-1, 0, 1), repeat=3) if any(o)]
    for i, coordinate in enumerate(coordinates):
        for offset in offsets:
            j = index.get(tuple(coordinate + offset))
            if j is not None and j > i:
                edges.append((i, j))
    if not edges:
        raise ValueError("Mesh skeleton has no edges")
    points = (
        np.asarray(origin_xyz_m)
        + coordinates[:, ::-1] * np.asarray(spacing_zyx_m)[::-1]
    )
    return CenterlineGraph(
        points, distance[tuple(coordinates.T)], np.asarray(edges, dtype=np.int64)
    )
