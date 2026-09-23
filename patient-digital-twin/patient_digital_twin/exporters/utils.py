# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CT orientation and attenuation utilities for patient-twin bundles."""

import json
from dataclasses import dataclass
from pathlib import Path

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


INTERVENTIONAL_POINTS = (
    (-1000.0, 0.0),
    (-300.0, 0.0),
    (100.0, 0.0008),
    (300.0, 0.0028),
    (500.0, 0.006),
    (900.0, 0.009),
    (1500.0, 0.012),
    (3000.0, 0.02),
    (8000.0, 0.044),
)


def hu_to_mu(hu):
    """Interpolate the workflow's interventional curve (mm^-1), clamping tails."""
    knots = np.asarray(INTERVENTIONAL_POINTS)
    return np.asarray(np.interp(hu, knots[:, 0], knots[:, 1]), dtype=np.float32)


def save_attenuation(ct, output, *, source):
    """Write attenuation and the existing workflow metadata contract."""
    hu = ct.hu_zyx
    if hu.ndim != 3 or not hu.size or not np.isfinite(hu).all():
        raise ValueError("Expected a non-empty, finite 3D CT volume")
    mu = hu_to_mu(hu)
    metadata = {
        "shape_zyx": list(hu.shape),
        "spacing_zyx_mm": list(ct.spacing_zyx_mm),
        "origin_xyz_mm": list(ct.origin_xyz_mm),
        "hu_range": [float(hu.min()), float(hu.max())],
        "mu_range": [float(mu.min()), float(mu.max())],
        "source": str(source),
        "hu_to_mu": {
            "preset": "interventional",
            "hu_min": INTERVENTIONAL_POINTS[0][0],
            "hu_max": INTERVENTIONAL_POINTS[-1][0],
            "mu_min": INTERVENTIONAL_POINTS[0][1],
            "mu_max": INTERVENTIONAL_POINTS[-1][1],
            "control_points": [list(p) for p in INTERVENTIONAL_POINTS],
        },
        "anatomical_frame": ct.anatomical_frame,
        "source_orientation": ct.source_orientation,
        "direction_row_major_3x3": list(ct.direction),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "mu_volume.npy", mu)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
