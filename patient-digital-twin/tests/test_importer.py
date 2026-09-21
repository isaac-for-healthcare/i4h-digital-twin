# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nibabel as nib
import numpy as np
import pytest
from patient_digital_twin import Kind, SegmentationImporter
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
    body = importer.to_human_body(strict=True)
    structure = body.structures["kidney_right"]
    assert structure.kind == Kind.ORGAN
    assert body.structures["heart"].vertices is None
    np.testing.assert_allclose(structure.vertices.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(structure.body_vertices, structure.world_vertices)
    inverse = np.linalg.inv(importer.affine_xyz_to_imaging_m)
    voxels = body.imaging_vertices("kidney_right") @ inverse[:3, :3].T + inverse[:3, 3]
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
    body = importer.to_human_body(configuration=path)
    assert body.structures["liver"].is_empty
    assert body.structures["liver"].faces is None
    assert body.structures["heart"].is_empty
    body.set_anatomy_enabled(True)
    assert not body.structures["liver"].is_empty
    assert body.structures["heart"].is_empty


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


def test_truncated_bone_end_not_a_joint():
    labels = np.zeros((40, 30, 30), dtype=np.uint8)
    labels[0:25, 10:14, 3:7] = 87
    labels[27:35, 10:14, 13:17] = 33
    importer = SegmentationImporter.from_array(
        labels, {"left humerus": 87, "vertebrae L5": 33}
    )
    landmarks = importer.extract_landmarks()
    assert "left_shoulder" in landmarks
    assert "left_elbow" not in landmarks


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
    first = SegmentationImporter.from_array(
        masks, labelmap, affine_xyz_to_imaging_m=affine
    ).to_human_body()
    shift = np.array([1.2, -0.8, 2.5])
    shifted = affine.copy()
    shifted[:3, 3] += shift
    second = SegmentationImporter.from_array(
        masks, labelmap, affine_xyz_to_imaging_m=shifted
    ).to_human_body(configuration={"anatomy": {"enabled": False}})
    second.set_anatomy_enabled(True)
    all_points = np.concatenate([s.body_vertices for s in first.structures.values()])
    np.testing.assert_allclose(all_points.min(0) + all_points.max(0), 0, atol=1e-12)
    np.testing.assert_allclose(
        second.body_to_imaging[:3, 3] - first.body_to_imaging[:3, 3], shift
    )
    for name, structure in first.structures.items():
        np.testing.assert_allclose(
            structure.body_vertices, second.structures[name].body_vertices, atol=1e-12
        )
        np.testing.assert_allclose(
            first.imaging_vertices(name) + shift, second.imaging_vertices(name)
        )
        assert not hasattr(structure, "local_to_imaging")
        assert not hasattr(structure, "classification")
        assert not hasattr(structure, "labels")


def test_imported_landmarks_use_the_same_body_frame_as_meshes():
    labels = np.zeros((40, 30, 30), dtype=np.uint8)
    labels[0:25, 10:14, 3:7] = 1
    labels[27:35, 10:14, 13:17] = 2
    importer = SegmentationImporter.from_array(
        labels, {1: "humerus_left", 2: "vertebrae_L5"}
    )
    measured = importer.extract_landmarks()
    body = importer.to_human_body()
    assert measured and measured.keys() == body.landmarks.keys()
    for name, point in body.landmarks.items():
        np.testing.assert_allclose(point + body.body_to_imaging[:3, 3], measured[name])


def test_oblique_bone_cut_on_nonprincipal_scan_face_is_not_an_elbow():
    z, y, x = np.indices((24, 30, 50))
    mask = np.zeros(z.shape, dtype=np.uint8)
    mask[
        (x >= 5) & (x < 40) & (y >= 6) & (y < 10) & (np.abs(z - (5 + 0.5 * x)) < 2)
    ] = 1
    mask[3:6, 12:15, 3:6] = 2
    importer = SegmentationImporter.from_array(
        mask, {1: "humerus_left", 2: "vertebrae_L5"}
    )
    landmarks = importer.extract_landmarks()
    assert "left_shoulder" in landmarks
    assert "left_elbow" not in landmarks
