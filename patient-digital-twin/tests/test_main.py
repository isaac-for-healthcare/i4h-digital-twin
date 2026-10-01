# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nibabel as nib
import numpy as np
import pytest

from patient_digital_twin import main
from patient_digital_twin.importers._common import segmentation_anatomy, selected_labels


def test_named_selection_uses_native_ids_and_limits_meshes():
    image = nib.Nifti1Image(np.full((5, 5, 8), 6, np.uint8), np.eye(4))
    anatomy = segmentation_anatomy(image, {6: "aorta", 12: "liver"}, names=["aorta"])
    assert set(anatomy.structures) == {"aorta"}
    assert not anatomy.structures["aorta"].is_empty
    assert selected_labels({6: "aorta", 12: "liver"}, ["aorta"]) == {6: "aorta"}
    with pytest.raises(ValueError, match="Unsupported"):
        selected_labels({12: "liver"}, ["aorta"])


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, "requires --input"),
        ({"classes": ["not_anatomy"]}, "Unknown"),
        ({"classes": []}, "at least one"),
        ({"source": "nvgenerate", "input": "ct.nii.gz"}, "omit --input"),
        ({"source": "nvgenerate", "modality": "MR"}, "CT only"),
    ],
)
def test_invalid_arguments_fail_before_model(options, message, tmp_path):
    args = dict(source="nvsegment", classes=["aorta"], output=tmp_path / "patient.usdc")
    args.update(options)
    with pytest.raises(ValueError, match=message):
        main.run_pipeline(**args)


def test_nvsegment_to_real_usd(monkeypatch, tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd

    ct = tmp_path / "ct.nii.gz"
    image = nib.Nifti1Image(np.full((9, 9, 16), 100, np.float32), np.eye(4))
    nib.save(image, ct)

    def segment(self, *, names):
        assert names == ("liver",)
        return segmentation_anatomy(
            nib.Nifti1Image(np.ones(image.shape, np.uint8), image.affine),
            {1: "liver"},
            names=names,
        )

    monkeypatch.setattr(main.NVSegmentImporter, "to_anatomy_collection", segment)
    output = main.run_pipeline(
        source="nvsegment",
        input=ct,
        classes=["liver"],
        output=tmp_path / "patient.usdc",
    )
    stage = Usd.Stage.Open(str(output))
    names = [p.GetCustomDataByKey("anatomy:name") for p in stage.Traverse()]
    assert "liver" in names and "aorta" not in names
