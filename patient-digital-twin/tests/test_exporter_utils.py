# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export helpers preserve physical CT placement and the workflow file contract."""

import itertools
import json

import nibabel as nib
import numpy as np
import pytest
from patient_digital_twin.exporters.utils import (
    CtVolume,
    hu_to_mu,
    load_nifti_hu,
    save_attenuation,
)


@pytest.mark.parametrize("axes", list(itertools.permutations(range(3))))
@pytest.mark.parametrize("signs", list(itertools.product((-1, 1), repeat=3)))
def test_ct_orientation_preserves_every_voxel_position(tmp_path, axes, signs):
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    affine = np.eye(4)
    affine[:3, :3] = np.eye(3)[:, axes] * np.array(signs) * [1.5, 2.0, 3.5]
    affine[:3, 3] = [13, -21, 37]
    path = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    ct = load_nifti_hu(path)
    np.testing.assert_allclose(np.reshape(ct.direction, (3, 3)), np.eye(3))
    assert ct.source_orientation == "".join(nib.aff2axcodes(affine)[::-1])
    # Unique voxel values identify the original indices independently of any
    # orientation helper. Compare their physical locations before/after ingest.
    for zyx in np.ndindex(ct.hu_zyx.shape):
        source_xyz = np.unravel_index(int(ct.hu_zyx[zyx]), data.shape)
        expected = (affine @ [*source_xyz, 1])[:3] * [-1, -1, 1]
        actual = np.asarray(ct.origin_xyz_mm) + np.asarray(zyx[::-1]) * np.asarray(
            ct.spacing_zyx_mm[::-1]
        )
        np.testing.assert_allclose(actual, expected)


def test_ct_preserves_obliquity_and_applies_intensity_scaling_once(tmp_path):
    angle = 0.2
    affine = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    image = nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.int16), affine)
    image.header.set_slope_inter(2, -1024)
    path = tmp_path / "ct.nii.gz"
    nib.save(image, path)
    ct = load_nifti_hu(path)
    assert not np.allclose(np.reshape(ct.direction, (3, 3)), np.eye(3))
    np.testing.assert_array_equal(ct.hu_zyx, -1022)


def test_attenuation_curve_and_metadata_contract(tmp_path):
    hu = np.array([-1500, -1000, -300, 100, 200, 300, 8000, 9000], dtype=np.float32)
    expected = np.array(
        [0, 0, 0.0035, 0.0055, 0.006, 0.0065, 0.02, 0.02], dtype=np.float32
    )
    np.testing.assert_array_equal(hu_to_mu(hu), expected)
    ct = CtVolume(
        hu.reshape(2, 2, 2), (3, 2, 1), (10, 20, 30), tuple(np.eye(3).ravel()), "SAR"
    )
    save_attenuation(ct, tmp_path, source="ct.nii.gz")
    np.testing.assert_array_equal(np.load(tmp_path / "mu_volume.npy").ravel(), expected)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert set(meta) == {
        "shape_zyx",
        "spacing_zyx_mm",
        "origin_xyz_mm",
        "hu_range",
        "mu_range",
        "source",
        "hu_to_mu",
        "anatomical_frame",
        "source_orientation",
        "direction_row_major_3x3",
    }
    assert meta["shape_zyx"] == [2, 2, 2]
    assert meta["spacing_zyx_mm"] == [3, 2, 1]
    assert meta["origin_xyz_mm"] == [10, 20, 30]
    assert meta["hu_range"] == [-1500, 9000]
    assert meta["anatomical_frame"] == "LPS"
    assert meta["source_orientation"] == "SAR"
    assert meta["hu_to_mu"]["preset"] == "linear"
