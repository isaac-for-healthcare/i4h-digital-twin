# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional TotalSegmentator Python-API adapter for CT and MR images."""

from __future__ import annotations

from pathlib import Path

from ._common import catalog_labels, coverage, image_input, segmentation_anatomy


class TotalSegmentatorImporter:
    """Segment a NIfTI/path or XYZ array; install with pip install TotalSegmentator.

    CT uses `total`, MR uses `total_mr`. Request every catalog class that the
    selected model supports; unavailable labels remain empty in AnatomyCollection.
    Model weights are downloaded by the upstream API on first use.
    """

    def __init__(
        self, image, *, modality="CT", affine_xyz_to_imaging_m=None, device="gpu"
    ):
        self.image = image
        self.modality = modality.upper()
        if self.modality not in ("CT", "MR"):
            raise ValueError("modality must be CT or MR")
        self.affine = affine_xyz_to_imaging_m
        self.device = device
        self.report = None

    def to_anatomy_collection(self, *, configuration=None):
        """Run the optional API, then return meshes in the shared body frame."""
        try:
            from totalsegmentator.map_to_binary import class_map
            from totalsegmentator.python_api import totalsegmentator
        except ImportError as exc:
            raise ImportError(
                "TotalSegmentator is optional; run: pip install TotalSegmentator"
            ) from exc
        task = "total" if self.modality == "CT" else "total_mr"
        labelmap = class_map[task]
        supported = catalog_labels(labelmap)
        image = image_input(self.image, self.affine)
        segmentation = totalsegmentator(
            image,
            task=task,
            roi_subset=[labelmap[i] for i in supported],
            ml=True,
            device=self.device,
        )
        body = segmentation_anatomy(segmentation, labelmap, configuration=configuration)
        body.source_path = (
            str(self.image) if isinstance(self.image, (str, Path)) else None
        )
        self.report = coverage(
            body, supported.values(), backend="TotalSegmentator", task=task
        )
        return body
