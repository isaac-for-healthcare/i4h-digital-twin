# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each exported vessel is one physics region with a literature stiffness default."""

import numpy as np
import pytest
import yaml
from patient_digital_twin import HumanBody, SegmentationImporter
from patient_digital_twin.anatomy import CATALOG, Kind, canonical_name, is_vessel
from patient_digital_twin.vessels import (
    ARTERY_DEFAULT,
    SOURCES,
    VEIN_DEFAULT,
    VESSEL_STIFFNESS,
    vessel_stiffness,
)


def test_stiffness_table_names_are_catalog_vessels_with_sources():
    for name, (modulus, source) in VESSEL_STIFFNESS.items():
        assert is_vessel(name), name
        assert modulus > 0 and source in SOURCES
    assert CATALOG["femoral_artery_left"][0] == Kind.VESSEL
    assert canonical_name("Right Femoral Artery") == "femoral_artery_right"
    assert CATALOG["portal_vein_and_splenic_vein"][0] == Kind.GROUP  # Existing entries keep their kind.


def test_stiffness_defaults_and_fallbacks():
    assert vessel_stiffness("aorta").young_modulus_kpa == 400.0
    assert vessel_stiffness("iliac_artery_left").young_modulus_kpa == 400.0
    assert vessel_stiffness("femoral_artery_right").young_modulus_kpa == 800.0
    assert vessel_stiffness("inferior_vena_cava").young_modulus_kpa == 23.0
    assert vessel_stiffness("splenic_artery") == ARTERY_DEFAULT
    assert vessel_stiffness("gonadal_vein_left") == VEIN_DEFAULT


def test_new_model_vessel_labels_are_imported():
    labels = np.zeros((12, 8, 8), np.uint8)
    labels[2:10, 3:6, 3:6] = 7
    anatomy = SegmentationImporter(labels, {7: "Left femoral artery"}).to_anatomy_collection()
    assert anatomy.structures["femoral_artery_left"].kind == Kind.VESSEL
    assert not anatomy.structures["femoral_artery_left"].is_empty


def _y_bundle(tmp_path, names):
    """Aorta (label 1) splitting into two iliac arteries (labels 2, 3), with a matching CT."""
    pytest.importorskip("vtk")
    pytest.importorskip("pxr")
    from patient_digital_twin.scan_volume import from_array

    labels = np.zeros((40, 20, 31), np.uint8)  # ZYX on a 1 mm grid.
    labels[20:37, 8:13, 13:18] = 1
    for z in range(5, 21):
        offset = (21 - z) // 2
        labels[z, 8:13, 13 - offset:18 - offset][labels[z, 8:13, 13 - offset:18 - offset] == 0] = 2
        labels[z, 8:13, 13 + offset:18 + offset][labels[z, 8:13, 13 + offset:18 + offset] == 0] = 3
    labelmap = {1: "aorta", 2: "iliac_artery_right", 3: "iliac_artery_left"}
    body = HumanBody(SegmentationImporter(labels, labelmap).to_anatomy_collection())
    body.attach_scan(from_array(np.where(labels, 300.0, 40.0).transpose(2, 1, 0), np.eye(4), world_unit="mm"))
    manifest = body.export_patient_twin(tmp_path / "bundle", vessel_names=names)
    return manifest, yaml.safe_load(manifest.read_text())


def test_bundle_labels_each_centerline_edge_with_its_vessel(tmp_path):
    names = ["aorta", "iliac_artery_right", "iliac_artery_left"]
    manifest, data = _y_bundle(tmp_path, names)
    folder = manifest.parent
    assert data["artifacts"]["centerline_regions"] == "centerline_regions.npy"
    assert data["artifacts"]["vessel_regions"] == "vessel_regions.yaml"
    edges = np.load(folder / "centerline_edges.npy")
    regions = np.load(folder / "centerline_regions.npy")
    assert regions.dtype == np.int32 and regions.shape == (len(edges),)
    assert set(regions.tolist()) == {0, 1, 2}
    mask = np.load(folder / "vessel_mask.npy")
    assert set(np.unique(mask).tolist()) == {0, 1}  # The navigation mask stays binary.

    # One connected graph across the junction, so the physics tree stays whole.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    count = len(np.load(folder / "centerline_points.npy"))
    graph = coo_matrix((np.ones(len(edges)), tuple(edges.T)), shape=(count, count))
    assert connected_components(graph, directed=False)[0] == 1

    # Below the bifurcation, edges belong to the branch they lie in (label 2 at lower X).
    points = np.load(folder / "centerline_points.npy")
    first = points[edges[:, 0]]
    low = first[:, 2] < 15
    assert (regions[low & (first[:, 0] > 16)] == 2).all() and (regions[low & (first[:, 0] < 14)] == 1).all()
    assert (regions[first[:, 2] > 24] == 0).all()

    sidecar = yaml.safe_load((folder / "vessel_regions.yaml").read_text())
    assert sidecar == {
        "schema_version": 1,
        "default": "unassigned",
        "branches": {0: "aorta", 1: "iliac_artery_right", 2: "iliac_artery_left"},
        "young_modulus_kpa": {"aorta": 400.0, "iliac_artery_right": 400.0, "iliac_artery_left": 400.0},
    }
    assert "wang_parker_2004" in (folder / "vessel_regions.yaml").read_text()


def test_overlapping_voxels_go_to_the_first_listed_vessel(tmp_path):
    manifest, _ = _y_bundle(tmp_path, ["iliac_artery_left", "aorta"])
    regions = np.load(manifest.parent / "centerline_regions.npy")
    sidecar = yaml.safe_load((manifest.parent / "vessel_regions.yaml").read_text())
    assert sidecar["branches"] == {0: "iliac_artery_left", 1: "aorta"}
    assert set(regions.tolist()) == {0, 1}


def test_write_artifacts_accepts_a_label_volume(tmp_path):
    pytest.importorskip("scipy")
    from patient_digital_twin.export import write_artifacts
    from patient_digital_twin.scan_volume import from_array

    labels = np.zeros((10, 10, 30), np.uint8)  # IJK order for from_array.
    labels[4:7, 4:7, 2:15] = 1
    labels[4:7, 4:7, 15:28] = 2
    ct = from_array(np.where(labels, 300.0, 40.0), np.eye(4), world_unit="mm")
    folder = write_artifacts(ct, tmp_path / "arrays", vessel_mask=labels, vessel_names=["aorta", "inferior_vena_cava"])
    regions = np.load(folder / "centerline_regions.npy")
    first = np.load(folder / "centerline_points.npy")[np.load(folder / "centerline_edges.npy")[:, 0]]
    np.testing.assert_array_equal(regions, (first[:, 2] >= 15).astype(np.int32))
    sidecar = yaml.safe_load((folder / "vessel_regions.yaml").read_text())
    assert sidecar["young_modulus_kpa"] == {"aorta": 400.0, "inferior_vena_cava": 23.0}

    binary = write_artifacts(ct, tmp_path / "binary", vessel_mask=labels > 0)
    assert not (binary / "centerline_regions.npy").exists() and not (binary / "vessel_regions.yaml").exists()
    with pytest.raises(ValueError, match="labels"):
        write_artifacts(ct, tmp_path / "bad", vessel_mask=labels)
