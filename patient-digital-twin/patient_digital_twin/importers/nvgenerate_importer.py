# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NV-Generate-CTMR paired CT and segmentation generation."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from ._common import (
    backend_context,
    coverage,
    runtime,
    segmentation_anatomy,
    selected_labels,
)
from ._segmentation import nifti_affine_m


class NVGenerateImporter:
    """Generate paired CT and anatomy using upstream rflow-ct inference.

    Install patient-digital-twin[nvgenerate] and provide the upstream checkout,
    mask/image checkpoints and conditioning dataset. Inference uses imports in
    the current process; python_executable opts into a separate environment.
    All generated labels survive the upstream output filtering step. After
    import, ct_scan holds the matching native CT for HumanBody.attach_scan().
    """

    def __init__(self, *, source_root=None, python_executable=None):
        self.root = Path(
            source_root or os.environ.get("NV_GENERATE_ROOT", ".")
        ).resolve()
        self.python_executable = python_executable
        self.report = None
        self.seed = None
        self.ct_scan = None

    def to_anatomy_collection(self, *, names=None):
        """Return meshes and retain the matching CT array/affine on this importer."""
        self.ct_scan = None
        python = runtime(
            self.python_executable,
            ["torch", "monai", "einops", "huggingface_hub", "matplotlib"],
            "patient-digital-twin[nvgenerate]",
        )
        if not (self.root / "scripts/inference.py").is_file():
            raise ImportError(
                "NV-Generate-CTMR source is missing. Clone https://github.com/NVIDIA-Medtech/NV-Generate-CTMR and supply source_root; pip install -r <source_root>/requirements.txt in the selected Python environment."
            )
        labels = json.loads((self.root / "configs/label_dict.json").read_text())
        labelmap = {value: name for name, value in labels.items()}
        supported = selected_labels(labelmap, names)
        self.seed = secrets.randbits(32)
        with TemporaryDirectory(prefix="patient-nvgenerate-") as temp:
            output = Path(temp) / "mask.nii.gz"
            ct_output = Path(temp) / "ct.nii.gz"
            if python is None:
                from ._nvgenerate_worker import generate

                with backend_context(self.root):
                    generate(self.seed, output, ct_output)
            else:
                worker = Path(__file__).with_name("_nvgenerate_worker.py").read_text()
                subprocess.run(
                    [python, "-c", worker, str(self.seed), output, ct_output],
                    cwd=self.root, check=True,
                )
            mask_image, ct_image = nib.load(output), nib.load(ct_output)
            if mask_image.shape != ct_image.shape or not np.allclose(
                nifti_affine_m(mask_image), nifti_affine_m(ct_image)
            ):
                raise ValueError(
                    "Generated CT and segmentation must share their physical grid"
                )
            from ..scan_volume import from_nifti

            # ScanVolume rejects non-finite intensities.
            scan = from_nifti(ct_output)
            scan.metadata["source"] = {
                "kind": "generated",
                "backend": "NV-Generate-CTMR",
                "seed": self.seed,
            }
            body = segmentation_anatomy(
                mask_image, labelmap, names=names
            )
        self.ct_scan = scan
        self.report = coverage(
            body,
            supported.values(),
            backend="NV-Generate-CTMR",
            requested=names,
            seed=self.seed,
        )
        return body
