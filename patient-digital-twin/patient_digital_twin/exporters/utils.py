# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CT orientation and attenuation utilities for patient-twin bundles."""

from dataclasses import dataclass

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class CtVolume:
    """HU voxels in ZYX order, with XYZ LPS spatial metadata in millimetres."""

    hu_zyx: np.ndarray
    spacing_zyx_mm: tuple[float, float, float]
    origin_xyz_mm: tuple[float, float, float]
    direction: tuple[float, ...]
    source_orientation: str
    anatomical_frame: str = "LPS"


def load_nifti_hu(path):
    """Permute/flip a NIfTI into LPS without resampling its HU values.

    Preserve any residual oblique rotation so callers can reject unsupported
    acquisitions. NIfTI scaling is applied by get_fdata, exactly once.
    """
    image = nib.load(str(path))
    return _image_to_ct(image)


def attached_ct(imaging):
    """Convert attached CT voxels to the workflow LPS layout, without file I/O."""
    if imaging.modality != "CT":
        raise ValueError("This exporter requires CT imaging in Hounsfield units")
    affine_mm = imaging.voxel_to_imaging.copy()
    affine_mm[:3] *= 1000
    image = nib.Nifti1Image(
        imaging.volume.transpose(2, 1, 0).astype(np.float32), affine_mm
    )
    return _image_to_ct(image)


def _image_to_ct(image):
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D CT volume, got shape {image.shape}")
    affine = np.asarray(image.affine)
    if not np.isfinite(affine).all() or np.linalg.matrix_rank(affine[:3, :3]) < 3:
        raise ValueError("CT affine must be finite and non-degenerate")
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(affine),
        nib.orientations.axcodes2ornt(("L", "P", "S")),
    )
    data = nib.orientations.apply_orientation(
        image.get_fdata().astype(np.float32), orientation
    )
    lps_affine = (
        np.diag([-1.0, -1.0, 1.0, 1.0])
        @ affine
        @ nib.orientations.inv_ornt_aff(orientation, image.shape)
    )
    spacing = np.linalg.norm(lps_affine[:3, :3], axis=0)
    return CtVolume(
        hu_zyx=np.ascontiguousarray(data.transpose(2, 1, 0)),
        spacing_zyx_mm=tuple(float(v) for v in spacing[::-1]),
        origin_xyz_mm=tuple(float(v) for v in lps_affine[:3, 3]),
        direction=tuple(float(v) for v in (lps_affine[:3, :3] / spacing).ravel()),
        source_orientation="".join(nib.aff2axcodes(affine)[::-1]),
    )
