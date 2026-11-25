# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-export quality check repairs simple mesh errors and bridges centerline gaps."""

import numpy as np
import pytest
import yaml
from patient_digital_twin import HumanBody, SegmentationImporter
from patient_digital_twin.geometry import mask_to_mesh
from patient_digital_twin.quality import bridge_label_gaps, components, repair_mesh
from patient_digital_twin.vessels import circulation

TETRA_V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
TETRA_F = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])


def test_clean_and_malformed_meshes_are_left_untouched():
    vertices, faces, fixes = repair_mesh(TETRA_V, TETRA_F)
    assert vertices is TETRA_V and faces is TETRA_F and fixes == {}
    bad = np.array([[0, 1, 9]])
    assert repair_mesh(TETRA_V, bad)[1] is bad  # Export's validation reports out-of-range indices.


def test_repair_mesh_fixes_simple_errors():
    # STL-style: every face has its own vertex copies (0-11), then a NaN vertex (12), an unused vertex
    # (13), a point on the edge from vertex 0 to 1 (14) and a point off the tetrahedron (15).
    vertices = np.vstack([TETRA_V[TETRA_F].reshape(-1, 3), [[np.nan, 0, 0], [5, 5, 5], [0.5, 0, 0], [0, 0, -1]]])
    faces = np.vstack([
        np.arange(12).reshape(4, 3),
        [[12, 0, 1]],     # non-finite vertex
        [[0, 0, 1]],      # repeated index
        [[0, 2, 14]],     # collinear: zero area
        [[0, 1, 2]],      # the first face again, same winding
        [[0, 2, 15]],     # with [0, 15, 2]: a back-to-back pair
        [[0, 15, 2]],
    ]).astype(np.int32)
    v, f, fixes = repair_mesh(vertices.astype(np.float32), faces)
    assert fixes == {"non_finite_faces": 1, "welded_vertices": 8, "degenerate_faces": 2, "back_to_back_faces": 2,
                     "duplicate_faces": 1, "unreferenced_vertices": 4}
    assert v.dtype == np.float32 and f.dtype == np.int32 and len(v) == 4 and len(f) == 4
    np.testing.assert_allclose(np.sort(v, axis=0), np.sort(TETRA_V, axis=0))
    with pytest.raises(ValueError, match="no valid faces"):
        repair_mesh(TETRA_V, np.array([[0, 0, 1]]))


def test_repair_mesh_drops_small_islands():
    mask = np.zeros((20, 20, 20), np.uint8)
    mask[2:14, 2:14, 2:14] = 1
    mask[18, 18, 18] = 1  # A stray voxel: well under 1% of the faces.
    vertices, faces = mask_to_mesh(mask)
    v, f, fixes = repair_mesh(vertices, faces)
    assert fixes["island_faces"] > 0 and "unreferenced_vertices" in fixes
    assert v.max() < 15  # Only the block remains.
    v2, f2, fixes2 = repair_mesh(vertices, faces, min_island_fraction=0.0)
    assert "island_faces" not in fixes2


def test_circulations():
    assert circulation("aorta") == circulation("femoral_artery_left") == "systemic_artery"
    assert circulation("inferior_vena_cava") == circulation("iliac_vena_left") == "systemic_vein"
    assert circulation("portal_vein_and_splenic_vein") == "portal"
    assert circulation("pulmonary_vein") == "pulmonary_vein" and circulation("pulmonary_artery") == "pulmonary_artery"


def test_bridge_label_gaps_paints_a_tube_within_one_circulation():
    labels = np.zeros((30, 12, 12), np.uint8)
    labels[2:12, 4:9, 4:9] = 1  # Five voxels wide.
    labels[15:28, 5:8, 5:8] = 2  # Three wide, three empty voxels further on (a 4-voxel gap).
    labels[2:12, 4:9, 10:11] = 3  # Touches nothing but is 1 voxel from label 1: a different circulation.
    out, bridges = bridge_label_gaps(labels, (1.0, 1.0, 1.0), 4.0, circulation=[0, 0, 0, 1])
    assert bridges == [(2, 1, 4.0)]
    from scipy import ndimage

    assert ndimage.label(np.isin(out, [1, 2]), np.ones((3, 3, 3)))[1] == 1  # One connected piece...
    assert set(np.unique(out[12:15])) == {0, 2}  # ...through a tube carrying the joining label.
    assert (out[labels > 0] == labels[labels > 0]).all()  # Vessels are never overwritten.
    assert not (out[:, :, 9] == 3).any() and (out[:, :, 9] == 0).all()  # Label 3 is never joined.
    assert bridge_label_gaps(labels, (1.0, 1.0, 1.0), 3.9, circulation=[0, 0, 0, 1])[1] == []
    assert bridge_label_gaps(labels, (2.0, 1.0, 1.0), 4.0)[1] == [(3, 1, 2.0)]  # One group: 3 joins first.


def _gap_body(labelmap):
    pytest.importorskip("vtk")
    pytest.importorskip("pxr")
    from patient_digital_twin.scan_volume import from_array

    labels = np.zeros((40, 20, 40), np.uint8)  # ZYX, 1 mm voxels.
    labels[5:35, 8:13, 8:13] = 1  # A vertical vessel...
    labels[18:23, 8:13, 16:35] = 2  # ...and a side branch whose label stops 3 mm short of it.
    body = HumanBody(SegmentationImporter(labels, labelmap).to_anatomy_collection())
    body.attach_scan(from_array(np.where(labels, 300.0, 40.0).transpose(2, 1, 0), np.eye(4), world_unit="mm"))
    return body, labels


def test_bundle_export_bridges_a_segmentation_gap(tmp_path):
    body, labels = _gap_body({1: "aorta", 2: "subclavian_artery_left"})
    manifest = body.export_patient_twin(tmp_path / "bundle", vessel_names=["aorta", "subclavian_artery_left"])
    folder, data = manifest.parent, yaml.safe_load(manifest.read_text())
    edges, points = np.load(folder / "centerline_edges.npy"), np.load(folder / "centerline_points.npy")
    regions = np.load(folder / "centerline_regions.npy")
    assert components(edges, len(points)) == 1 and data["quality"]["centerline_components"] == 1
    assert data["quality"]["centerline_bridges"] == [{"from": "subclavian_artery_left", "to": "aorta", "gap_mm": 4.0}]
    assert data["quality"]["mesh_repairs"] == {}
    gap = (points[edges[:, 0], 0] > 12.5) & (points[edges[:, 0], 0] < 15.5)  # Edges starting inside the gap.
    assert gap.any() and (regions[gap] == 1).all()  # The bridge belongs to the joining vessel.
    assert np.load(folder / "vessel_mask.npy").sum() == (labels > 0).sum()  # The exported mask is not filled.

    raw = body.export_patient_twin(tmp_path / "raw", vessel_names=["aorta", "subclavian_artery_left"], repair_body=False)
    assert "quality" not in yaml.safe_load(raw.read_text())
    assert components(np.load(raw.parent / "centerline_edges.npy"), len(np.load(raw.parent / "centerline_points.npy"))) == 2


@pytest.mark.parametrize("other", ["iliac_vena_left", "portal_vein_and_splenic_vein"])
def test_vessels_of_different_circulations_never_join(tmp_path, other):
    body, _ = _gap_body({1: "inferior_vena_cava" if other.startswith("portal") else "aorta", 2: other})
    names = list(body.anatomy.structures)
    quality = yaml.safe_load(body.export_patient_twin(tmp_path / "bundle", vessel_names=names).read_text())["quality"]
    assert quality["centerline_bridges"] == [] and quality["centerline_components"] == 2


def test_body_repair_fixes_meshes_in_place():
    mask = np.zeros((20, 20, 20), np.uint8)
    mask[2:14, 2:14, 2:14] = 1
    mask[18, 18, 18] = 1
    body = HumanBody(SegmentationImporter(mask, {1: "aorta"}).to_anatomy_collection())
    report = body.repair()
    assert report["mesh_repairs"]["aorta"]["island_faces"] > 0
    assert body.repair() == {"mesh_repairs": {}}  # Idempotent.
