# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the bundled example's grid, label map and completed generation record."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def test_high_resolution_example_pair():
    root = Path(__file__).parents[1] / "examples/data/nv_ct_high_resolution"
    if not (root / "ct.nii.gz").exists():
        pytest.skip("Generated example data is not present in this checkout")
    mask = nib.load(root / "segmentation.nii.gz")
    ct = nib.load(root / "ct.nii.gz")
    assert mask.shape == ct.shape == (512, 512, 768)
    np.testing.assert_allclose(mask.affine, ct.affine, atol=1e-4)
    np.testing.assert_allclose(mask.header.get_zooms(), [0.763, 0.763, 0.7875])
    assert mask.header.get_xyzt_units()[0] == ct.header.get_xyzt_units()[0] == "mm"
    assert mask.get_data_dtype() == np.dtype("uint8")
    assert ct.get_data_dtype() == np.dtype("int16")
    labels = json.loads((root / "labels.json").read_text())
    assert labels["body"] == 200 and labels["background"] == 0
    sample = np.asarray(mask.dataobj[:, :, 384])
    assert set(np.unique(sample)) <= set(labels.values())
    metadata = json.loads((root / "provenance.json").read_text())
    assert metadata["status"] == "complete"
    assert set(metadata["sha256"]) == {
        "ct.nii.gz",
        "segmentation.nii.gz",
        "labels.json",
    }
