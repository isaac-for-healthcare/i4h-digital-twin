# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nibabel as nib
import numpy as np
import pytest
from patient_digital_twin import HumanBody, Kind, SegmentationImporter
from patient_digital_twin.geometry import transform_points
from patient_digital_twin.importers._segmentation import (
    canonical_name,
    normalize_labelmap,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("left rib 12", "rib_left_12"),
        ("vertebrae T12", "vertebrae_T12"),
        ("left lung upper lobe", "lung_upper_lobe_left"),
        ("bladder", "urinary_bladder"),
        ("humerus-L", "humerus_left"),
        ("left kidney cyst", "kidney_cyst_left"),
    ],
)
def test_nv_generate_names(raw, expected):
    assert canonical_name(raw) == expected


def test_full_affine_and_local_mesh_roundtrip(tmp_path):
    labels = np.zeros((10, 12, 14), dtype=np.uint8)
    labels[0:4, 3:7, 4:8] = 5  # Touches scan boundary, still must be closed.
    affine = np.array(
        [[-2, 0.3, 0, 80], [0, 0, -4, 120], [0, 3, 0, -70], [0, 0, 0, 1.0]]
    )
    image = nib.Nifti1Image(labels, affine)
    image.header.set_xyzt_units("mm")
    path = tmp_path / "sample_label.nii.gz"
    nib.save(image, path)
    importer = SegmentationImporter(path, {"right kidney": 5, "heart": 115})
    np.testing.assert_array_equal(importer.masks_zyx, labels.transpose(2, 1, 0))
    body = HumanBody(importer.to_anatomy_collection(strict=True))
    structure = body.anatomy.structures["kidney_right"]
    assert structure.kind == Kind.ORGAN
    assert body.anatomy.structures["heart"].vertices is None
    np.testing.assert_allclose(structure.vertices.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(structure.body_vertices, structure.world_vertices)
    inverse = np.linalg.inv(importer.affine_xyz_to_imaging_m)
    voxels = (
        transform_points(structure.body_vertices, body.anatomy.body_to_imaging)
        @ inverse[:3, :3].T
        + inverse[:3, 3]
    )
    np.testing.assert_allclose(voxels.min(0), [-0.5, 2.5, 3.5], atol=1e-5)
    np.testing.assert_allclose(voxels.max(0), [3.5, 6.5, 7.5], atol=1e-5)
    triangles = structure.vertices[structure.faces]
    volume = (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6
    )
    edges = np.concatenate(
        [
            structure.faces[:, [0, 1]],
            structure.faces[:, [1, 2]],
            structure.faces[:, [2, 0]],
        ]
    )
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert (counts == 2).all() and volume > 0


def test_nifti_units(tmp_path):
    labels = np.ones((3, 3, 3), np.uint8)
    for units, spacing in (("meter", 0.002), ("mm", 2), ("micron", 2000)):
        image = nib.Nifti1Image(labels, np.diag([spacing] * 3 + [1]))
        image.header.set_xyzt_units(units)
        path = tmp_path / (units + ".nii")
        nib.save(image, path)
        importer = SegmentationImporter(path, {"liver": 1})
        np.testing.assert_allclose(
            importer.affine_xyz_to_imaging_m, np.diag([0.002] * 3 + [1])
        )


def test_importer_accepts_anatomy_configuration(tmp_path):
    path = tmp_path / "anatomy.yaml"
    path.write_text("anatomy:\n  enabled: false\n", encoding="utf-8")
    importer = SegmentationImporter.from_array(
        np.ones((3, 3, 3)), {1: "liver", 2: "heart"}
    )
    body = HumanBody(importer.to_anatomy_collection(configuration=path))
    assert body.anatomy.structures["liver"].is_empty
    assert body.anatomy.structures["liver"].faces is None
    assert body.anatomy.structures["heart"].is_empty
    body.anatomy.set_enabled(True)
    assert not body.anatomy.structures["liver"].is_empty
    assert body.anatomy.structures["heart"].is_empty


def test_invalid_input_and_unknown_labels():
    with pytest.raises(ValueError, match="integer"):
        SegmentationImporter.from_array(np.full((3, 3, 3), 1.5), {1: "liver"})
    with pytest.raises(ValueError, match="missing"):
        SegmentationImporter.from_array(np.ones((3, 3, 3)), {2: "liver"})
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_labelmap({"liver": 1, "heart": 1})


def test_directory_grid_and_overlap_rejected(tmp_path):
    image = nib.Nifti1Image(np.ones((3, 3, 3), np.uint8), np.eye(4))
    nib.save(image, tmp_path / "liver.nii.gz")
    nib.save(image, tmp_path / "spleen.nii.gz")
    with pytest.raises(ValueError, match="Overlapping"):
        SegmentationImporter(tmp_path)
    image.set_sform(np.diag([2, 2, 2, 1]))
    nib.save(image, tmp_path / "spleen.nii.gz")
    with pytest.raises(ValueError, match="mismatch"):
        SegmentationImporter(tmp_path)


def test_body_origin_is_shared_and_independent_of_scan_translation_and_visibility():
    masks = np.zeros((12, 14, 16), dtype=np.uint8)
    masks[1:4, 2:5, 3:6] = 1
    masks[5:10, 7:12, 10:14] = 2
    labelmap = {1: "liver", 2: "kidney_left"}
    affine = np.array(
        [
            [-0.002, 0.0003, 0, 0.12],
            [0, 0, -0.004, -0.08],
            [0, 0.003, 0, 0.25],
            [0, 0, 0, 1],
        ]
    )
    first = HumanBody(
        SegmentationImporter.from_array(
            masks, labelmap, affine_xyz_to_imaging_m=affine
        ).to_anatomy_collection()
    )
    shift = np.array([1.2, -0.8, 2.5])
    shifted = affine.copy()
    shifted[:3, 3] += shift
    second = HumanBody(
        SegmentationImporter.from_array(
            masks, labelmap, affine_xyz_to_imaging_m=shifted
        ).to_anatomy_collection(configuration={"anatomy": {"enabled": False}})
    )
    second.anatomy.set_enabled(True)
    all_points = np.concatenate(
        [s.body_vertices for s in first.anatomy.structures.values()]
    )
    np.testing.assert_allclose(all_points.min(0) + all_points.max(0), 0, atol=1e-12)
    np.testing.assert_allclose(
        second.anatomy.body_to_imaging[:3, 3] - first.anatomy.body_to_imaging[:3, 3],
        shift,
    )
    for name, structure in first.anatomy.structures.items():
        np.testing.assert_allclose(
            structure.body_vertices,
            second.anatomy.structures[name].body_vertices,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            transform_points(structure.body_vertices, first.anatomy.body_to_imaging)
            + shift,
            transform_points(
                second.anatomy.structures[name].body_vertices,
                second.anatomy.body_to_imaging,
            ),
        )
        assert not hasattr(structure, "local_to_imaging")
        assert not hasattr(structure, "classification")
        assert not hasattr(structure, "labels")
