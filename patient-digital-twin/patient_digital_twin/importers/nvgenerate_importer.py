# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NV-Generate-CTMR mask diffusion only; no image-generation model is loaded."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib

from ._common import catalog_labels, coverage, run_backend, runtime, segmentation_body


class NVGenerateImporter:
    """Generate a fresh CT anatomical mask using the native mask-model defaults.

    Requires an NV-Generate-CTMR checkout, its two mask checkpoints, and optional
    pip-installed runtime requirements. The model's native grid is 256 cubed,
    1.5 mm isotropic, with 1000 DDPM steps. All supported catalog labels survive;
    anatomy is not restricted to the upstream example's single requested organ.
    """

    def __init__(self, *, source_root=None, python_executable=None):
        self.root = Path(
            source_root or os.environ.get("NV_GENERATE_ROOT", ".")
        ).resolve()
        self.python_executable = python_executable
        self.report = None
        self.seed = None

    def to_human_body(self, *, configuration=None):
        """Sample with a fresh seed for every call and return a HumanBody."""
        python = runtime(
            self.python_executable, ["torch", "monai", "einops"], "torch monai einops"
        )
        if not (self.root / "scripts/sample_mask.py").is_file():
            raise ImportError(
                "NV-Generate-CTMR source is missing. Clone https://github.com/NVIDIA-Medtech/NV-Generate-CTMR and supply source_root; pip install -r <source_root>/requirements.txt in the selected Python environment."
            )
        labels = json.loads((self.root / "configs/label_dict.json").read_text())
        labelmap = {value: name for name, value in labels.items()}
        supported = catalog_labels(labelmap)
        previous = self.seed
        while self.seed is None or self.seed == previous:
            self.seed = secrets.randbits(32)
        worker = Path(__file__).with_name("_nvgenerate_worker.py").read_text()
        with TemporaryDirectory(prefix="patient-nvgenerate-") as temp:
            output = Path(temp) / "mask.nii.gz"
            run_backend([python, "-c", worker, str(self.seed), output], cwd=self.root)
            body = segmentation_body(
                nib.load(output), labelmap, configuration=configuration
            )
        self.report = coverage(
            body, supported.values(), backend="NV-Generate-CTMR", seed=self.seed
        )
        return body
