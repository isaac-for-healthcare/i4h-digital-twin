# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``vdt-segment-vessels --ts-gt-dir`` writes the label mask in the CT cache's axes (nvbugs 7043583)."""

from __future__ import annotations

import warnings

import nibabel as nib
import numpy as np
import pytest
from vasculature_digital_twin.cli import preprocess_ct, segment_vessels

# Stored in RAS (nibabel's identity affine), so the default LPS cache flips the two in-plane axes.
_AFFINE = np.diag([0.8, 0.8, 1.0, 1.0])


def _write_case(tmp_path):
    """A CT with one off-centre vessel (bright) and its TotalSegmentator label, stored in RAS."""
    shape_xyz = (32, 32, 40)
    xx, yy, zz = np.ogrid[: shape_xyz[0], : shape_xyz[1], : shape_xyz[2]]
    # a tube along z, off-centre in x and y, so a missing flip moves it
    vessel = ((xx - 9) ** 2 + (yy - 21) ** 2 <= 9) & (zz >= 4) & (zz < 36)
    ct = np.where(vessel, 400.0, -900.0).astype(np.float32)
    nib.save(nib.Nifti1Image(ct, _AFFINE), tmp_path / "ct.nii.gz")
    gt = tmp_path / "segmentations"
    gt.mkdir()
    nib.save(nib.Nifti1Image(vessel.astype(np.uint8), _AFFINE), gt / "aorta.nii.gz")
    return tmp_path / "ct.nii.gz", gt


@pytest.mark.parametrize("reorient", ["--reorient", "--no-reorient"])
def test_ts_gt_mask_is_in_the_cache_frame(tmp_path, reorient):
    ct, gt = _write_case(tmp_path)
    cache = tmp_path / "cache"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert preprocess_ct.main(["--nifti", str(ct), "--output-dir", str(cache), reorient]) == 0
        assert (
            segment_vessels.main(
                ["--ct-dir", str(cache), "--ts-gt-dir", str(gt), "--ts-labels", "aorta", "--close-iterations", "0"]
            )
            == 0
        )

    hu = np.load(cache / "hu_volume.npy")
    mask = np.load(cache / "vessel_mask.npy")
    # The label marks exactly the bright voxels, so in the cache's axes the two must coincide.
    np.testing.assert_array_equal(mask > 0, hu > 0)
