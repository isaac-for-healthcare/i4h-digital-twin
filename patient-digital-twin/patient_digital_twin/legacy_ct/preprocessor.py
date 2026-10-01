# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .config import HuToMuMapping, PreprocessingSettings
from .ct.dicom_ingest import CtVolume, load_dicom_series_hu, load_nifti_hu
from .hu_mapping import hu_to_mu
from .volume import PreprocessedVolume, VolumeMetadata


class VolumePreprocessor:
    """Convert CT HU volumes into mu volumes plus metadata."""

    def __init__(
        self,
        hu_volume: np.ndarray,
        spacing_zyx_mm: tuple[float, float, float],
        origin_xyz_mm: tuple[float, float, float] | None = None,
        source: str | None = None,
        settings: PreprocessingSettings | None = None,
        anatomical_frame: str | None = None,
        source_orientation: str | None = None,
        direction: tuple[float, ...] | None = None,
    ):
        if hu_volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {hu_volume.shape}")

        self._hu_volume = hu_volume.astype(np.float32, copy=False)
        self._spacing_zyx_mm = spacing_zyx_mm
        self._origin_xyz_mm = origin_xyz_mm
        self._source = source
        self._settings = settings or PreprocessingSettings()
        self._anatomical_frame = anatomical_frame
        self._source_orientation = source_orientation
        self._direction = direction

    @classmethod
    def _from_ct_volume(
        cls,
        ct: CtVolume,
        source: str,
        settings: PreprocessingSettings | None,
    ) -> "VolumePreprocessor":
        return cls(
            hu_volume=ct.hu_zyx,
            spacing_zyx_mm=ct.spacing_zyx_mm or (1.0, 1.0, 1.0),
            origin_xyz_mm=ct.origin_xyz_mm,
            source=source,
            settings=settings,
            anatomical_frame=ct.anatomical_frame,
            source_orientation=ct.source_orientation,
            direction=ct.direction,
        )

    @classmethod
    def from_dicom(
        cls,
        dicom_dir: str | Path,
        settings: PreprocessingSettings | None = None,
        reorient: bool = True,
    ) -> "VolumePreprocessor":
        """Load a DICOM series, reoriented into the canonical LPS frame by default."""
        dicom_dir = Path(dicom_dir)
        if not dicom_dir.exists():
            raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
        ct = load_dicom_series_hu(dicom_dir, reorient=reorient)
        return cls._from_ct_volume(ct, source=str(dicom_dir), settings=settings)

    @classmethod
    def from_nifti(
        cls,
        nifti_path: str | Path,
        settings: PreprocessingSettings | None = None,
        reorient: bool = True,
    ) -> "VolumePreprocessor":
        """Load a NIfTI file, reoriented into the canonical LPS frame by default."""
        nifti_path = Path(nifti_path)
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
        ct = load_nifti_hu(nifti_path, reorient=reorient)
        return cls._from_ct_volume(ct, source=str(nifti_path), settings=settings)

    @classmethod
    def from_numpy(
        cls,
        hu_volume: np.ndarray,
        spacing_zyx_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        settings: PreprocessingSettings | None = None,
        anatomical_frame: str | None = None,
    ) -> "VolumePreprocessor":
        """Wrap an in-memory HU volume.

        Args:
            hu_volume: HU volume in ``(axis 0, axis 1, axis 2)`` array order.
            spacing_zyx_mm: Voxel spacing in mm matching the volume axes.
            settings: Preprocessing settings. Defaults to :class:`PreprocessingSettings`.
            anatomical_frame: Frame the caller guarantees the axes are already in, e.g.
                ``"LPS"``. Left unresolved by default, since a bare array carries no
                orientation metadata.
        """
        return cls(
            hu_volume=hu_volume,
            spacing_zyx_mm=spacing_zyx_mm,
            settings=settings,
            anatomical_frame=anatomical_frame,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._hu_volume.shape

    @property
    def hu_volume_zyx(self) -> np.ndarray:
        return self._hu_volume

    @property
    def hu_range(self) -> tuple[float, float]:
        return (float(self._hu_volume.min()), float(self._hu_volume.max()))

    @property
    def settings(self) -> PreprocessingSettings:
        return self._settings

    @property
    def anatomical_frame(self) -> str | None:
        return self._anatomical_frame

    def with_hu_to_mu(self, mapping: HuToMuMapping) -> "VolumePreprocessor":
        """Return a preprocessor with a different transfer function, same HU volume.

        Lets a window/level sweep re-run :meth:`preprocess` without reloading the CT.

        Args:
            mapping: Transfer function to use instead of the current one.

        Returns:
            New preprocessor sharing this instance's HU volume and metadata.
        """
        return VolumePreprocessor(
            hu_volume=self._hu_volume,
            spacing_zyx_mm=self._spacing_zyx_mm,
            origin_xyz_mm=self._origin_xyz_mm,
            source=self._source,
            settings=dataclasses.replace(self._settings, hu_to_mu=mapping),
            anatomical_frame=self._anatomical_frame,
            source_orientation=self._source_orientation,
            direction=self._direction,
        )

    def preprocess(self, output_dir: str | Path | None = None) -> PreprocessedVolume:
        settings = self._settings
        hu = self._hu_volume
        hu_range = (float(hu.min()), float(hu.max()))

        if settings.clip_hu:
            hu = np.clip(hu, settings.hu_clip_min, settings.hu_clip_max)

        mu = hu_to_mu(hu, settings.hu_to_mu)
        mu_range = (float(mu.min()), float(mu.max()))

        metadata = VolumeMetadata(
            shape_zyx=mu.shape,
            spacing_zyx_mm=self._spacing_zyx_mm,
            origin_xyz_mm=self._origin_xyz_mm,
            hu_range=hu_range,
            mu_range=mu_range,
            source=self._source,
            hu_to_mu=settings.hu_to_mu.to_dict(),
            anatomical_frame=self._anatomical_frame,
            source_orientation=self._source_orientation,
            direction=self._direction,
        )
        volume = PreprocessedVolume(mu, metadata)

        if output_dir is not None:
            volume.save(output_dir)
            print(f"[VolumePreprocessor] Saved to: {output_dir}")
        return volume
