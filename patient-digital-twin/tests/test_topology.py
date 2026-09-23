# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Topology selection, local-frame storage, and VTK/VMTK integration contracts."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from patient_digital_twin import (
    AnatomicalStructure,
    CenterlineGraph,
    HumanBody,
    Kind,
    topology,
)
from patient_digital_twin.catalog import CATALOG


def graph():
    return CenterlineGraph(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
        np.array([0.01, 0.01]),
        np.array([[0, 1]]),
    )


def test_body_extracts_tubular_anatomy_including_disabled_and_composite(monkeypatch):
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    names = {
        "aorta": Kind.VESSEL,
        "trachea": CATALOG["trachea"],
        "bronchus": Kind.AIRWAY,
        "portal_vein_and_splenic_vein": Kind.GROUP,
        "liver": Kind.ORGAN,
        "lung_left": Kind.ORGAN,
    }
    body = HumanBody(
        {
            n: AnatomicalStructure(n, k, vertices.copy(), np.array([[0, 1, 2]]))
            for n, k in names.items()
        }
    )
    body.anatomy.structures["aorta"].enabled = False
    body.anatomy.structures["empty"] = AnatomicalStructure("empty", Kind.VESSEL)
    calls = []

    def extract(v, f):
        calls.append((v, f))
        return graph()

    monkeypatch.setattr(topology, "extract_centerlines", extract)
    result = body.extract_topology()
    assert set(result) == {
        "aorta",
        "trachea",
        "bronchus",
        "portal_vein_and_splenic_vein",
    }
    assert len(calls) == 4
    structure = body.anatomy.structures["aorta"]
    assert structure.centerline is result["aorta"]
    structure.local_to_world[:3, 3] = 1
    np.testing.assert_array_equal(structure.centerline.points, graph().points)
    structure.enabled = True
    assert structure.centerline is result["aorta"]
    structure.vertices = vertices.copy()
    assert structure.centerline is None
    structure.centerline = graph()
    structure.faces = np.array([[0, 2, 1]])
    assert structure.centerline is None


def test_failure_is_named_and_does_not_commit_partial_results(monkeypatch):
    body = HumanBody(
        {
            n: AnatomicalStructure(
                n,
                Kind.VESSEL,
                np.ones((3, 3)),
                np.array([[0, 1, 2]]),
                centerline=graph(),
            )
            for n in ["first", "second"]
        }
    )
    previous = body.anatomy.structures["first"].centerline

    def extract(v, f):
        if v is body.anatomy.structures["second"].mesh.vertices:
            raise ValueError("bad tube")
        return graph()

    monkeypatch.setattr(topology, "extract_centerlines", extract)
    with pytest.raises(RuntimeError, match="second: bad tube"):
        body.extract_topology()
    assert body.anatomy.structures["first"].centerline is previous


def test_missing_optional_runtime(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "vmtk", None)
    with pytest.raises(ImportError, match="optional VTK and VMTK"):
        topology.extract_centerlines([], [])
    assert HumanBody().extract_topology() == {}


def _cylinder(vtk, offset=0):
    from vtk.util.numpy_support import vtk_to_numpy

    source = vtk.vtkCylinderSource()
    source.SetRadius(0.01)
    source.SetHeight(0.1)
    source.SetResolution(24)
    source.SetCenter(offset, 0, 0)
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputConnection(source.GetOutputPort())
    triangulate.Update()
    mesh = triangulate.GetOutput()
    return (
        vtk_to_numpy(mesh.GetPoints().GetData()).copy(),
        vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1, 4)[:, 1:].copy(),
    )


def test_vtk_components_polylines_and_radii(monkeypatch):
    vtk = pytest.importorskip("vtk")
    surfaces = []

    class Extractor:
        def Execute(self):
            assert self.UseJoblib is False
            surfaces.append(self.Surface)
            points = vtk.vtkPoints()
            for p in [(0, -0.04, 0), (0, 0, 0), (0, 0.04, 0)]:
                points.InsertNextPoint(*p)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(3)
            for i in range(3):
                lines.InsertCellPoint(i)
            radius = vtk.vtkDoubleArray()
            radius.SetName("MaximumInscribedSphereRadius")
            for _ in range(3):
                radius.InsertNextValue(0.01)
            self.Centerlines = vtk.vtkPolyData()
            self.Centerlines.SetPoints(points)
            self.Centerlines.SetLines(lines)
            self.Centerlines.GetPointData().AddArray(radius)

    monkeypatch.setattr(
        topology,
        "_runtime",
        lambda: (vtk, SimpleNamespace(vmtkCenterlinesNetwork=Extractor)),
    )
    a, f = _cylinder(vtk)
    b, g = _cylinder(vtk, 0.2)
    vertices = np.concatenate([a, b])
    original = vertices.copy()
    result = topology.extract_centerlines(vertices, np.concatenate([f, g + len(a)]))
    assert len(surfaces) == 2
    np.testing.assert_array_equal(result.edges, [[0, 1], [1, 2], [3, 4], [4, 5]])
    np.testing.assert_allclose(result.radii, 0.01)
    np.testing.assert_array_equal(vertices, original)
    with pytest.raises(ValueError, match="triangle indices"):
        topology.extract_centerlines(a, [[0, 1, len(a)]])
    with pytest.raises(ValueError, match="no MaximumInscribedSphereRadius"):
        output = vtk.vtkPolyData()
        output.SetPoints(surfaces[0].GetPoints())
        topology._graph(output)


@pytest.mark.skipif(
    importlib.util.find_spec("vmtk") is None, reason="Optional VMTK not installed"
)
def test_real_vmtk_capped_tube():
    import vtk

    vertices, faces = _cylinder(vtk)
    result = topology.extract_centerlines(vertices, faces)
    assert len(result.edges) > 0
    assert np.ptp(result.points[:, 1]) > 0.05
    assert np.max(np.linalg.norm(result.points[:, [0, 2]], axis=1)) < 0.006
    assert np.median(result.radii) == pytest.approx(0.01, rel=0.3)


def test_mesh_skeleton_grid_round_trip_and_physical_radii():
    pytest.importorskip("vtk")
    from patient_digital_twin.imaging_to_mesh import mask_to_mesh
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    mask = np.zeros((24, 17, 17), dtype=bool)
    mask[2:22, 6:11, 6:11] = True
    spacing = np.array([0.002, 0.001, 0.0015])
    origin = np.array([-0.02, 0.3, -0.1])
    vertices, faces = mask_to_mesh(
        mask, spacing_zyx_mm=spacing * 1000, origin_xyz_mm=origin * 1000
    )
    grid = {"shape_zyx": mask.shape, "spacing_zyx_m": spacing, "origin_xyz_m": origin}
    recovered = topology.voxelize_mesh(vertices * 0.001, faces, **grid)
    np.testing.assert_array_equal(recovered, mask)
    result = topology.extract_centerlines(
        vertices * 0.001, faces, method="skeleton", **grid
    )
    coordinates = np.argwhere(skeletonize(mask))
    np.testing.assert_allclose(
        result.points, origin + coordinates[:, ::-1] * spacing[::-1]
    )
    np.testing.assert_allclose(
        result.radii,
        distance_transform_edt(mask, sampling=spacing)[tuple(coordinates.T)],
    )
    assert len(result.edges) == len(result.points) - 1
    with pytest.raises(ValueError, match="Grid needs"):
        topology.voxelize_mesh(
            vertices * 0.001, faces, **{**grid, "spacing_zyx_m": [0, 1, 1]}
        )


def test_topology_selection_preserves_other_centerlines(monkeypatch):
    body = HumanBody(
        {
            n: AnatomicalStructure(
                n,
                Kind.VESSEL,
                np.ones((3, 3)),
                np.array([[0, 1, 2]]),
                centerline=graph(),
            )
            for n in ["aorta", "vascular_tree"]
        }
    )
    original = body.anatomy.structures["aorta"].centerline
    monkeypatch.setattr(topology, "extract_centerlines", lambda v, f: graph())
    assert set(body.extract_topology(names=["vascular_tree"])) == {"vascular_tree"}
    assert body.anatomy.structures["aorta"].centerline is original
    with pytest.raises(KeyError, match="Unknown anatomy"):
        body.extract_topology(names=["missing"])
