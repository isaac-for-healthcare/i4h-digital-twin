# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An explicitly attached image volume and its physical registration."""

from dataclasses import dataclass

import numpy as np

from .geometry import rigid_transform


@dataclass(frozen=True)
class ImagingVolume:
    """Owned ZYX voxels with XYZ voxel-to-RAS and body-to-RAS transforms in meters.

    CT values must already be Hounsfield units. source_path is optional provenance,
    never an instruction to load a file. Arrays are copied and made read-only.
    """

    volume: np.ndarray
    voxel_to_imaging: np.ndarray
    body_to_imaging: np.ndarray
    source_path: str | None = None
    modality: str = "CT"
    source_scan: object | None = None

    def __post_init__(self):
        if not isinstance(self.volume, np.ndarray):
            raise TypeError("Imaging must be provided as a NumPy volume")
        if (
            self.volume.ndim != 3
            or not self.volume.size
            or self.volume.dtype.kind not in "iuf"
            or not np.isfinite(self.volume).all()
        ):
            raise ValueError(
                "Imaging must be a non-empty, finite, real numeric 3D ZYX volume"
            )
        affine = np.array(self.voxel_to_imaging, dtype=float, copy=True)
        if (
            affine.shape != (4, 4)
            or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or np.linalg.matrix_rank(affine[:3, :3]) < 3
        ):
            raise ValueError("voxel_to_imaging must be an invertible affine in meters")
        registration = rigid_transform(self.body_to_imaging).copy()
        volume = self.volume.copy()
        for name, value in (
            ("volume", volume),
            ("voxel_to_imaging", affine),
            ("body_to_imaging", registration),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not isinstance(self.modality, str) or not self.modality.strip():
            raise ValueError("modality must be a non-empty string")
        object.__setattr__(self, "modality", self.modality.upper())
        if self.source_path is not None:
            object.__setattr__(self, "source_path", str(self.source_path))
