# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional NV-Segment-CTMR MONAI-bundle adapter (CT_BODY / MRI_BODY)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from ._segmentation import SegmentationImporter

from ._common import (
    coverage,
    image_input,
    run_backend,
    runtime,
    segmentation_anatomy,
    selected_labels,
)


class NVSegmentImporter:
    """Run the upstream bundle in its own working directory and Python runtime.

    NV-Segment-CTMR is a source bundle, not a standalone PyPI distribution.
    Supply its inner NV-Segment-CTMR directory and pip-install the upstream
    runtime requirements in python_executable (the current Python by default).
    """

    def __init__(
        self,
        image,
        *,
        bundle_root=None,
        modality="CT",
        affine_xyz_to_imaging_m=None,
        python_executable=None,
    ):
        self.image = image
        self.root = Path(
            bundle_root or os.environ.get("NV_SEGMENT_CTMR_ROOT", ".")
        ).resolve()
        self.modality = modality.upper()
        if self.modality not in ("CT", "MR"):
            raise ValueError("modality must be CT or MR")
        self.affine = affine_xyz_to_imaging_m
        self.python_executable = python_executable
        self.report = None

    def to_anatomy_collection(self, *, configuration=None, names=None):
        """Request supported catalog prompts, invert preprocessing, and mesh output."""
        python = runtime(
            self.python_executable,
            ["torch", "monai", "ignite", "fire", "einops"],
            "monai pytorch-ignite fire einops huggingface_hub",
        )
        config = self.root / "configs/inference.json"
        if not config.is_file():
            raise ImportError(
                "NV-Segment-CTMR source bundle is missing. Clone https://github.com/NVIDIA-Medtech/NV-Segment-CTMR and set bundle_root to its inner NV-Segment-CTMR directory; pip-install its runtime requirements."
            )
        definitions = json.loads((self.root / "configs/label_dict.json").read_text())
        dataset = "CT" if self.modality == "CT" else "MRI"
        labelmap = {
            item["index"]: name
            for name, item in definitions.items()
            if dataset in item.get("datasets", [])
        }
        supported = selected_labels(labelmap, names)
        if not supported:
            raise ValueError(
                f"No catalog labels supported by NV-Segment for {self.modality}"
            )
        with TemporaryDirectory(prefix="patient-nvsegment-") as temp:
            temp = Path(temp)
            source = temp / "image.nii.gz"
            input_image = image_input(self.image, self.affine)
            nib.save(input_image, source)
            overrides = json.loads(config.read_text())
            overrides.update(
                bundle_root=str(self.root),
                input_dict={"image": str(source)},
                modality="CT_BODY" if self.modality == "CT" else "MRI_BODY",
                everything_labels=list(supported),
                output_dir=str(temp / "masks"),
                output_postfix="segmentation",
                separate_folder=False,
            )
            config_path = temp / "inference.json"
            config_path.write_text(json.dumps(overrides))
            run_backend(
                [
                    python,
                    "-m",
                    "monai.bundle",
                    "run",
                    "--config_file",
                    config_path,
                    "--meta_file",
                    self.root / "configs/metadata.json",
                ],
                cwd=self.root,
            )
            outputs = list((temp / "masks").rglob("*.nii.gz"))
            if len(outputs) != 1:
                raise RuntimeError(
                    f"Expected one NV-Segment output, found {len(outputs)}"
                )
            result = nib.load(outputs[0])
            if result.shape != input_image.shape or not np.allclose(
                SegmentationImporter._affine_m(result),
                SegmentationImporter._affine_m(input_image),
                atol=1e-6,
            ):
                raise ValueError(
                    "NV-Segment output must match the input image physical grid"
                )
            # VistaPostTransformd restores prompt IDs before saving the NIfTI.
            body = segmentation_anatomy(
                result, supported, configuration=configuration, names=names
            )
        body.source_path = (
            str(self.image) if isinstance(self.image, (str, Path)) else None
        )
        self.report = coverage(
            body,
            supported.values(),
            backend="NV-Segment-CTMR",
            requested=names,
            modality=self.modality,
        )
        return body
