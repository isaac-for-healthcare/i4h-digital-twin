# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CT ingest from DICOM series and NIfTI files into a consistent in-memory convention."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .orientation import CANONICAL_FRAME, affine_to_lps, to_canonical_lps


@dataclass(frozen=True)
class CtVolume:
    """A CT volume in a consistent in-memory convention.

    Attributes:
        hu_zyx: HU volume in ``(axis 0, axis 1, axis 2)`` array order.
        spacing_zyx_mm: Voxel spacing in mm matching the axes of ``hu_zyx``.
        origin_xyz_mm: Patient-space (LPS) position of voxel ``[0, 0, 0]``.
        direction: Row-major 3x3 direction cosines in LPS, columns in index order
            ``(i, j, k)``, i.e. array axes ``(2, 1, 0)``.
        anatomical_frame: Frame the axes are expressed in, ``"LPS"`` once reoriented (see
            :mod:`vasculature_digital_twin.ct.orientation`), or None if unresolved.
        source_orientation: Anatomical directions the source array axes increased toward
            before reorientation, e.g. ``"IPL"``.
    """

    hu_zyx: np.ndarray
    spacing_zyx_mm: tuple[float, float, float] | None = None
    origin_xyz_mm: tuple[float, float, float] | None = None
    direction: tuple[float, ...] | None = None
    anatomical_frame: str | None = None
    source_orientation: str | None = None

    def to_json_dict(self) -> dict:
        data: dict = {
            "shape_zyx": list(self.hu_zyx.shape),
            "dtype": str(self.hu_zyx.dtype),
        }
        if self.spacing_zyx_mm is not None:
            data["spacing_zyx_mm"] = list(self.spacing_zyx_mm)
        if self.origin_xyz_mm is not None:
            data["origin_xyz_mm"] = list(self.origin_xyz_mm)
        if self.direction is not None:
            data["direction_row_major_3x3"] = list(self.direction)
        if self.anatomical_frame is not None:
            data["anatomical_frame"] = self.anatomical_frame
        if self.source_orientation is not None:
            data["source_orientation"] = self.source_orientation
        return data


def _canonicalize(
    hu_zyx: np.ndarray,
    direction: tuple[float, ...],
    spacing_zyx_mm: tuple[float, float, float],
    origin_xyz_mm: tuple[float, float, float],
    source: str,
) -> CtVolume:
    """Reorient a freshly loaded volume into the canonical frame."""
    canonical = to_canonical_lps(
        hu_zyx=hu_zyx,
        direction=direction,
        spacing_zyx_mm=spacing_zyx_mm,
        origin_xyz_mm=origin_xyz_mm,
    )
    if canonical.is_oblique:
        warnings.warn(
            f"{source} is an oblique acquisition ({canonical.max_obliquity_deg:.1f} deg from the "
            f"nearest patient axes). Axes were permuted and flipped to the closest canonical "
            f"orientation, but the residual rotation remains; resample the volume if exact "
            f"anatomical alignment is required.",
            stacklevel=3,
        )
    return CtVolume(
        hu_zyx=canonical.hu_zyx,
        spacing_zyx_mm=canonical.spacing_zyx_mm,
        origin_xyz_mm=canonical.origin_xyz_mm,
        direction=canonical.direction,
        anatomical_frame=CANONICAL_FRAME,
        source_orientation=canonical.source_code,
    )


def load_dicom_series_hu(dicom_dir: str | Path, reorient: bool = True) -> CtVolume:
    """Load a DICOM series directory into a HU volume.

    Voxels are already in Hounsfield Units on return: SimpleITK's GDCM reader applies the
    modality LUT (Rescale Slope / Rescale Intercept) while reading, so applying it again
    here would offset every volume by the intercept, typically -1024 HU.

    Args:
        dicom_dir: Directory containing the series.
        reorient: Reorient the volume into the canonical LPS frame using the series
            direction cosines. Disable only to inspect the acquisition as stored.

    Returns:
        The HU volume with its spacing, origin and orientation metadata.

    Raises:
        FileNotFoundError: If the directory does not exist.
        RuntimeError: If SimpleITK is unavailable or no series is found.
    """
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "SimpleITK is required to load DICOM series. Install it with:\n" "  pip install SimpleITK"
        ) from exc

    ddir = Path(dicom_dir)
    if not ddir.exists() or not ddir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {ddir}")

    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(str(ddir)))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found under: {ddir}")

    series_uid = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(str(ddir), series_uid)
    reader.SetFileNames(file_names)
    image = reader.Execute()

    arr_zyx = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)

    spacing_xyz = tuple(float(x) for x in image.GetSpacing())
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    origin_xyz = tuple(float(x) for x in image.GetOrigin())
    direction = tuple(float(x) for x in image.GetDirection())

    if not reorient:
        return CtVolume(
            hu_zyx=arr_zyx,
            spacing_zyx_mm=spacing_zyx,
            origin_xyz_mm=origin_xyz,
            direction=direction,
        )

    return _canonicalize(
        hu_zyx=arr_zyx,
        direction=direction,
        spacing_zyx_mm=spacing_zyx,
        origin_xyz_mm=origin_xyz,
        source=str(ddir),
    )


def load_nifti_hu(nifti_path: str | Path, reorient: bool = True) -> CtVolume:
    """Load a NIfTI file into a HU volume.

    The affine is interpreted as RAS (the NIfTI convention) and converted to LPS so that
    NIfTI and DICOM inputs land in the same patient frame.

    Args:
        nifti_path: Path to a ``.nii`` or ``.nii.gz`` file.
        reorient: Reorient the volume into the canonical LPS frame using the affine.
            Disable only to inspect the volume as stored.

    Returns:
        The HU volume with its spacing, origin and orientation metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If nibabel is unavailable.
    """
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required to load NIfTI files. Install with:\n  pip install nibabel") from exc

    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    image = nib.load(nifti_path)
    arr_ijk = image.get_fdata().astype(np.float32)
    arr_zyx = np.transpose(arr_ijk, (2, 1, 0))

    direction, spacing_ijk, origin_xyz = affine_to_lps(image.affine)
    spacing_zyx = (spacing_ijk[2], spacing_ijk[1], spacing_ijk[0])

    if not reorient:
        return CtVolume(
            hu_zyx=np.ascontiguousarray(arr_zyx),
            spacing_zyx_mm=spacing_zyx,
            origin_xyz_mm=origin_xyz,
            direction=direction,
        )

    return _canonicalize(
        hu_zyx=arr_zyx,
        direction=direction,
        spacing_zyx_mm=spacing_zyx,
        origin_xyz_mm=origin_xyz,
        source=str(nifti_path),
    )
