# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API for medical segmentation to OpenUSD conversion."""

from .converter import (
    convert_mask_to_usd,
    convert_nrrd_to_nifti,
    convert_path,
    convert_segmentation_array,
    convert_segmentation_file,
    load_labelmap,
)
from .labels import CATEGORY_LABELS
from .mesh import mask_to_mesh, write_obj
from .models import ConversionResult, MeshArtifact
from .usd import write_usd

__all__ = [
    "CATEGORY_LABELS",
    "ConversionResult",
    "MeshArtifact",
    "convert_mask_to_usd",
    "convert_nrrd_to_nifti",
    "convert_path",
    "convert_segmentation_array",
    "convert_segmentation_file",
    "load_labelmap",
    "mask_to_mesh",
    "write_obj",
    "write_usd",
]

__version__ = "0.1.0"
