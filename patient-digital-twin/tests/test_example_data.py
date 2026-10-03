# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The original s0011 CT/masks are available through Git LFS and retain their grid."""

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def test_s0011_dataset():
    root = Path(__file__).parents[1] / "examples/data/s0011"
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["license"] == "CC-BY-4.0" and not provenance["modified"]
    ct = nib.load(root / "ct.nii.gz")  # An unresolved LFS pointer must fail, not skip.
    assert ct.shape == (311, 311, 431)
    for relative, expected_hash in provenance["sha256"].items():
        path = root / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        image = nib.load(path)
        assert image.shape == ct.shape
        np.testing.assert_allclose(image.affine, ct.affine)
    aorta = nib.load(root / "segmentations/aorta.nii.gz")
    assert set(np.unique(np.asanyarray(aorta.dataobj))) == {0, 1}


def test_sample_mask_selection():
    from patient_digital_twin import SegmentationImporter

    root = Path(__file__).parents[1] / "examples/data/s0011/segmentations"
    importer = SegmentationImporter(root, names=["aorta"])
    assert set(importer._names.values()) == {"aorta"}
    anatomy = importer.to_anatomy_collection()
    assert set(anatomy.structures) == {"aorta"}
    assert not anatomy.structures["aorta"].is_empty
