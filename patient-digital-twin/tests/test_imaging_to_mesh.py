# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Surface extraction is bundled and works without exporters or a USD runtime."""

import numpy as np
import pytest
from patient_digital_twin.imaging_to_mesh import mask_to_mesh


def test_boundary_mesh_is_closed_outward_and_uses_xyz_units():
    mask = np.ones((2, 3, 4), dtype=bool)
    vertices, faces = mask_to_mesh(
        mask, spacing_zyx_mm=(3.0, 2.0, 1.0), origin_xyz_mm=(10.0, 20.0, 30.0)
    )
    assert vertices.dtype == np.float32 and faces.dtype == np.int32
    np.testing.assert_allclose(vertices.min(0), [9.5, 19.0, 28.5])
    np.testing.assert_allclose(vertices.max(0), [13.5, 25.0, 34.5])
    assert faces.min() >= 0 and faces.max() < len(vertices)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert (counts == 2).all()
    triangles = vertices[faces] - vertices.mean(0)
    assert (
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        > 0
    )
    assert mask.all()  # Extraction must not mutate its input.


def test_voxel_output_matches_default_spacing_and_origin():
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[1:4, 2:5, 3:6] = 1
    default_vertices, default_faces = mask_to_mesh(mask)
    vertices, faces = mask_to_mesh(
        mask.astype(bool), spacing_zyx_mm=(1.0, 1.0, 1.0), origin_xyz_mm=(0.0, 0.0, 0.0)
    )
    np.testing.assert_array_equal(default_vertices, vertices)
    np.testing.assert_array_equal(default_faces, faces)
    np.testing.assert_allclose(vertices.min(0), [2.5, 1.5, 0.5])
    np.testing.assert_allclose(vertices.max(0), [5.5, 4.5, 3.5])


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((3, 3, 3)),
        np.ones((3, 3)),
        np.empty((0, 2, 3)),
        np.full((3, 3, 3), 2),
        np.full((3, 3, 3), 0.5),
        np.full((3, 3, 3), np.nan),
        np.full((3, 3, 3), -1),
    ],
)
def test_invalid_masks_rejected(mask):
    with pytest.raises(ValueError):
        mask_to_mesh(mask)


@pytest.mark.parametrize(
    "options",
    [
        {"spacing_zyx_mm": (0, 1, 1)},
        {"spacing_zyx_mm": (-1, 1, 1)},
        {"spacing_zyx_mm": (1, 1)},
        {"spacing_zyx_mm": (np.inf, 1, 1)},
        {"origin_xyz_mm": (1, 2)},
        {"origin_xyz_mm": (0, 0, np.nan)},
    ],
)
def test_invalid_geometry_arguments_rejected(options):
    with pytest.raises(ValueError):
        mask_to_mesh(np.ones((2, 2, 2)), **options)
