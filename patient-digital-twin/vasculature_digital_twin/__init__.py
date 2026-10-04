# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated public API for vasculature digital twin preprocessing.

``vasculature_digital_twin`` is kept for existing users only and will be removed in a
future release; importing it emits a ``DeprecationWarning``. It reorients CT into a
canonical LPS frame, maps HU to linear attenuation (``VolumePreprocessor``,
``HuToMuMapping``), and segments vessels with centerlines (``get_vessel_mask``,
``extract_centerlines``, ``extract_vessel_mesh``).

Replacements in ``patient_digital_twin``:

- Native CT without reorientation: ``patient_digital_twin.scan_volume.from_nifti`` /
  ``from_dicom`` (``ScanVolume`` keeps the source axes, frame, and units).
- Vessel meshes and centerlines: ``SegmentationImporter`` or ``NVSegmentImporter``,
  then ``HumanBody.extract_topology`` and ``export_patient_twin(vessel_names=...)``.
- HU to attenuation mapping now belongs to i4h-sensor-simulation.
"""

import warnings

from .config import HuToMuMapping, PreprocessingSettings
from .ct.orientation import (
    CANONICAL_FRAME,
    CanonicalVolume,
    affine_to_lps,
    orientation_code,
    to_canonical_lps,
)
from .hu_mapping import hu_to_mu, hu_to_mu_curve
from .preprocessor import VolumePreprocessor
from .vasculature import (
    TOTALSEG_CORONARY_LABEL,
    TOTALSEG_VESSEL_TERRITORY_MAP,
    CenterlineGraph,
    VesselSegmentationResult,
    apply_vessel_boost,
    build_contrast_volume,
    compute_arrival_map,
    ct_coords_to_voxel,
    extract_centerlines,
    extract_vessel_mesh,
    gamma_variate,
    get_vessel_mask,
    vessel_mask_from_hu,
    vessel_mask_from_totalsegmentator,
)
from .volume import PreprocessedVolume, VolumeMetadata

warnings.warn(
    "vasculature_digital_twin is deprecated and will be removed in a future release; "
    "use patient_digital_twin (scan_volume, SegmentationImporter/NVSegmentImporter, "
    "HumanBody.extract_topology, export_patient_twin) instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CANONICAL_FRAME",
    "CanonicalVolume",
    "HuToMuMapping",
    "PreprocessingSettings",
    "VolumePreprocessor",
    "PreprocessedVolume",
    "VolumeMetadata",
    "TOTALSEG_CORONARY_LABEL",
    "TOTALSEG_VESSEL_TERRITORY_MAP",
    "CenterlineGraph",
    "VesselSegmentationResult",
    "affine_to_lps",
    "apply_vessel_boost",
    "build_contrast_volume",
    "compute_arrival_map",
    "ct_coords_to_voxel",
    "extract_centerlines",
    "extract_vessel_mesh",
    "gamma_variate",
    "get_vessel_mask",
    "hu_to_mu",
    "hu_to_mu_curve",
    "orientation_code",
    "to_canonical_lps",
    "vessel_mask_from_hu",
    "vessel_mask_from_totalsegmentator",
]

__version__ = "0.1.0"
