# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``HumanBody`` facade: a patient's anatomy plus attached CT, with topology and export.

Main classes:

- ``HumanBody``: the user-facing object. Wraps an ``AnatomyCollection`` (``body.anatomy``,
  from ``anatomy.py``), attaches CT (``attach_scan`` / ``attach_imaging``), extracts
  vessel and airway centerlines (``extract_topology``), and exports a standalone USD
  (``export_to_usd``) or an i4h-workflows bundle (``export_patient_twin``).
- ``ImagingVolume``: what ``body.imaging`` holds once CT is attached: a native
  ``ScanVolume`` plus the rigid body-to-RAS-meter registration of the anatomy.

Typical use::

    body = HumanBody(SegmentationImporter(masks_dir, names=["aorta"]).to_anatomy_collection())
    body.extract_topology(names=["aorta"])
    body.attach_scan(scan_volume.from_nifti("ct.nii.gz"))
    body.export_patient_twin("bundle", vessel_names=["aorta"])
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .anatomy import AnatomicalStructure, AnatomyCollection, Kind, is_vessel
from .geometry import CenterlineGraph, extract_centerlines, rigid_transform
from .scan_volume import ScanVolume, from_array


@dataclass(frozen=True)
class ImagingVolume:
    """An attached native HU scan plus the rigid body-to-RAS-meter registration of the anatomy.

    Created by ``HumanBody.attach_scan`` / ``attach_imaging``; read it from ``body.imaging``.
    """

    scan: ScanVolume
    body_to_imaging: np.ndarray
    source_path: str | None = None

    @property
    def volume(self) -> np.ndarray:
        """Read-only KJI (ZYX) voxels in Hounsfield units."""
        return self.scan.values_kji

    @property
    def voxel_to_imaging(self) -> np.ndarray:
        """4x4 map from IJK (XYZ) voxel indices to RAS meters."""
        return self.scan.ijk_to_ras_m


class HumanBody:
    """A patient: ``anatomy`` (an ``AnatomyCollection``) and optional attached ``imaging``.

    Example:
        ``body = HumanBody(importer.to_anatomy_collection())``; ``HumanBody({"aorta": structure})``
        also accepts a plain structure dictionary.
    """

    def __init__(self, anatomy: AnatomyCollection | Mapping[str, AnatomicalStructure] | None = None) -> None:
        self.anatomy = anatomy if isinstance(anatomy, AnatomyCollection) else AnatomyCollection(
            None if anatomy is None else dict(anatomy)
        )
        self.imaging: ImagingVolume | None = None

    def attach_scan(
        self, scan: ScanVolume, *, body_to_imaging: ArrayLike | None = None, source_path: str | Path | None = None
    ) -> ImagingVolume:
        """Attach CT as a ``ScanVolume``, keeping its native array order, frame, and units.

        Args:
            scan: For example ``scan_volume.from_nifti("ct.nii.gz")`` on the segmentation's grid.
            body_to_imaging: Rigid body-to-RAS-meter registration; defaults to the
                importer's ``anatomy.body_to_imaging``.
            source_path: Optional provenance; its parent folder names bundle ``patient_id``.

        Returns:
            The new ``body.imaging``. On failure the previous attachment is kept.
        """
        registration = self.anatomy.body_to_imaging if body_to_imaging is None else body_to_imaging
        if registration is None:
            raise ValueError("attach_imaging requires body_to_imaging registration")
        if not isinstance(scan, ScanVolume):
            raise TypeError("attach_scan expects a ScanVolume")
        matrix = rigid_transform(registration)
        matrix.setflags(write=False)
        source = None if source_path is None else str(source_path)
        self.imaging = ImagingVolume(scan, matrix, source)
        return self.imaging

    def attach_imaging(
        self,
        volume: ArrayLike,
        *,
        voxel_to_imaging: ArrayLike,
        body_to_imaging: ArrayLike | None = None,
        source_path: str | Path | None = None,
    ) -> ImagingVolume:
        """Attach a raw KJI (ZYX) HU array; it is copied into a new RAS-meter ``ScanVolume``.

        Args:
            volume: Real numeric 3D array indexed ``[k, j, i]``.
            voxel_to_imaging: 4x4 map from IJK (XYZ) voxel indices to RAS meters.
            body_to_imaging / source_path: As for ``attach_scan``.
        """
        array, kji_to_ijk = np.asarray(volume), np.eye(4)
        if array.dtype.kind not in "iuf":
            raise ValueError("Imaging must be a real numeric NumPy volume")
        kji_to_ijk[:3, :3] = np.eye(3)[:, ::-1]
        affine = np.asarray(voxel_to_imaging, dtype=float) @ kji_to_ijk
        scan = from_array(array, affine, array_axes="kji", world_unit="m")
        return self.attach_scan(scan, body_to_imaging=body_to_imaging, source_path=source_path)

    def export_to_usd(self, path: str | Path) -> Path:
        """Write anatomy (current pose), stored centerlines, and attached CT to one USD file.

        ``path`` must end in .usd, .usda, or .usdc; an existing file is replaced. Requires
        ``usd-core``. Returns the resolved path. See ``export.export_to_usd``.
        """
        from .export import export_to_usd

        return export_to_usd(self, path)

    def export_patient_twin(
        self,
        output: str | Path,
        *,
        patient_id: str | None = None,
        vessel_names: Iterable[str] = (),
        world_from_patient_m: ArrayLike | None = None,
        ct_exterior: bool = False,
        skin_opacity: float = 0.15,
    ) -> Path:
        """Write a new schema-2 bundle directory for i4h-workflows; returns its ``patient_twin.yaml``.

        Args:
            output: New directory path (must not exist).
            vessel_names: Vessels to rasterize on the CT grid with navigation centerlines
                (requires attached CT).
            ct_exterior: Add a CT-derived skin envelope (requires attached CT).
            patient_id / world_from_patient_m / skin_opacity: Optional manifest metadata,
                simulator placement hint, and envelope opacity.

        Example:
            ``body.export_patient_twin("bundle", vessel_names=["aorta"], ct_exterior=True)``
        """
        from .export import export_patient_twin

        return export_patient_twin(
            self, output, patient_id=patient_id, vessel_names=tuple(vessel_names),
            world_from_patient_m=world_from_patient_m, ct_exterior=ct_exterior, skin_opacity=skin_opacity,
        )

    def extract_topology(
        self, *, names: Iterable[str] | None = None, spacing_m: float = 0.0015
    ) -> dict[str, CenterlineGraph]:
        """Compute and store centerlines for vessel and airway meshes (including disabled ones).

        Args:
            names: Structures to consider (default: all); non-tubular ones are skipped.
            spacing_m: Voxel size used to rasterize each mesh before skeletonizing.

        Returns:
            ``{name: CenterlineGraph}`` in each mesh's local frame, also stored on
            ``structure.centerline``. Nothing is stored unless every extraction succeeds.

        Example:
            ``body.extract_topology(names=["aorta"], spacing_m=0.0015)["aorta"].points``
        """
        if not np.isfinite(spacing_m) or spacing_m <= 0:
            raise ValueError("spacing_m must be positive and finite")
        selected = set(self.anatomy.structures) if names is None else set(names)
        if unknown := selected - self.anatomy.structures.keys():
            raise KeyError(f"Unknown anatomy: {sorted(unknown)}")
        results: dict[str, CenterlineGraph] = {}
        for name, structure in self.anatomy.structures.items():
            vertices, faces = structure.mesh.vertices, structure.mesh.faces
            tubular = is_vessel(name, structure.kind) or structure.kind == Kind.AIRWAY
            if name not in selected or not tubular or vertices is None or faces is None:
                continue
            origin = vertices.min(0) - 2 * spacing_m
            shape = np.ceil((vertices.max(0) - origin) / spacing_m).astype(int) + 3
            try:
                results[name] = extract_centerlines(
                    vertices, faces, shape_zyx=(int(shape[2]), int(shape[1]), int(shape[0])),
                    spacing_zyx_m=(spacing_m,) * 3, origin_xyz_m=origin,
                )
            except ImportError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Centerline extraction failed for {name}: {exc}") from exc
        for name, graph in results.items():
            self.anatomy.structures[name].centerline = graph
        return results
