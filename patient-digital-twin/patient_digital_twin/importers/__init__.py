# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional generation, segmentation, and mesh importers returning AnatomyCollection.

Heavy backend dependencies are loaded only when an importer is run.
"""

from ._segmentation import SegmentationImporter
from .nvgenerate_importer import NVGenerateImporter
from .nvsegment_importer import NVSegmentImporter
from .simple_importer import SimpleImporter

__all__ = [
    "NVGenerateImporter",
    "NVSegmentImporter",
    "SegmentationImporter",
    "SimpleImporter",
]
