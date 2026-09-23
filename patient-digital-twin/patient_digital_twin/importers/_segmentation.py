# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import NV-Generate-CTMR/MAISI NIfTI label volumes into a AnatomyCollection.

The label dictionary must match the generation configuration; numeric IDs
cannot safely be inferred from the image or TotalSegmentator's current map.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import nibabel as nib
import numpy as np

from ..geometry import transform_points
from ..anatomy import AnatomyCollection

SKIP_STRUCTURES = re.compile(r"^(background|body|dummy\d*|.*_trunc)$")


def canonical_name(raw: str) -> str:
    """Match NV-Generate names to the curated TotalSegmentator metadata."""
    parts = [part for part in re.split(r"[^a-z0-9]+", raw.lower()) if part]
    side = None
    for index, part in enumerate(parts):
        if part in ("left", "right", "l", "r"):
            side = {"l": "left", "r": "right"}.get(part, part)
            parts.pop(index)
            break
    name = "_".join(parts)
    if side:
        # Curated ribs put the side before the rib number.
        if re.fullmatch(r"rib_\d+", name):
            return f"rib_{side}_{name.split('_')[1]}"
        name += "_" + side
    name = {"bladder": "urinary_bladder"}.get(name, name)
    return re.sub(
        r"^vertebrae_([ctls])(\d+)$",
        lambda match: "vertebrae_" + match[1].upper() + match[2],
        name,
    )


def normalize_labelmap(value: Mapping | str | Path) -> dict[int, str]:
    """Accept NV-Generate name->ID JSON or an ID->name mapping."""
    if isinstance(value, (str, Path)):
        with Path(value).open() as stream:
            value = json.load(stream)
    if not isinstance(value, Mapping) or not value:
        raise ValueError("labelmap must be a non-empty name->ID or ID->name mapping")
    result = {}
    for key, item in value.items():
        if isinstance(item, str):
            try:
                label_id = int(key)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid label ID: {key!r}") from exc
            if str(label_id) != str(key):
                raise ValueError(f"Invalid label ID: {key!r}")
            name = item
        else:
            if not isinstance(item, (int, np.integer)) or isinstance(item, bool):
                raise TypeError(f"Invalid label ID: {item!r}")
            label_id, name = int(item), key
        if label_id < 0 or not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid label entry: {key!r}: {item!r}")
        if label_id in result:
            raise ValueError(f"Duplicate label ID {label_id}")
        result[label_id] = name
    return result


class SegmentationImporter:
    """One 3D integer mask volume indexed (slice Z, Y, X), plus its labelmap.

    A file is a multi-label NIfTI from NV-Generate-CTMR. A directory is one
    binary NIfTI per structure (TotalSegmentator); overlapping masks are
    rejected because a single integer volume cannot represent overlaps.
    Full NIfTI affine orientation, shear, reflection and spatial units are
    preserved. Missing NIfTI units default to millimeters, as in MAISI.
    """

    def __init__(self, path: str | Path, labelmap: Mapping | str | Path | None = None):
        """Read a multi-label NIfTI plus matching map, or a binary-mask directory.

        Reading validates masks and coordinates but does not mesh them. Call
        to_anatomy_collection() to extract surfaces. Directory names define their labels.
        """
        path = Path(path)
        self.source_path = path
        if path.is_dir():
            self._load_directory(path)
        else:
            if labelmap is None:
                raise ValueError(
                    "Pass the matching NV-Generate configs/label_dict.json (or CTMR map)"
                )
            image = nib.load(str(path))
            affine = self._affine_m(image)
            self._initialize(
                np.asanyarray(image.dataobj).transpose(2, 1, 0),
                normalize_labelmap(labelmap),
                affine,
            )

    @staticmethod
    def _affine_m(image) -> np.ndarray:
        """Normalize a 3D NIfTI voxel affine to XYZ meters without losing orientation."""
        if len(image.shape) != 3:
            raise ValueError(f"Expected a 3D NIfTI, got {image.shape}")
        unit = image.header.get_xyzt_units()[0]
        factors = {"unknown": 0.001, "mm": 0.001, "meter": 1.0, "micron": 1e-6}
        if unit not in factors:
            raise ValueError(f"Unsupported NIfTI spatial unit: {unit}")
        affine = image.affine.copy()
        affine[:3] *= factors[unit]
        return affine

    def _load_directory(self, path):
        """Merge binary masks on one grid; reject overlaps rather than overwrite labels."""
        files = sorted(path.glob("*.nii")) + sorted(path.glob("*.nii.gz"))
        if not files:
            raise FileNotFoundError(f"No NIfTI masks in {path}")
        reference = nib.load(str(files[0]))
        affine = self._affine_m(reference)
        volume = np.zeros(reference.shape[::-1], dtype=np.int32)
        labels = {}
        for label_id, file in enumerate(files, 1):
            image = nib.load(str(file))
            if image.shape != reference.shape or not np.allclose(
                self._affine_m(image), affine, atol=1e-8, rtol=1e-7
            ):
                raise ValueError(f"Mask grid/affine mismatch: {file}")
            name = file.name.removesuffix(".gz").removesuffix(".nii")
            if SKIP_STRUCTURES.fullmatch(canonical_name(name)):
                continue
            data = np.asanyarray(image.dataobj)
            if not np.isfinite(data).all() or not np.isin(data, [0, 1]).all():
                raise ValueError(f"Expected a binary mask: {file}")
            mask = data.transpose(2, 1, 0).astype(bool)
            if np.any(volume[mask] != 0):
                raise ValueError(
                    f"Overlapping binary masks cannot form a label volume: {file}"
                )
            volume[mask] = label_id
            labels[label_id] = name
        self._initialize(volume, labels, affine)

    @classmethod
    def from_array(cls, masks_zyx, labelmap, *, affine_xyz_to_imaging_m=None):
        """Copy a nonnegative integer (Z, Y, X) volume and its matching label map.

        affine_xyz_to_imaging_m maps XYZ voxel indices to imaging-world meters;
        omitted affines use 1 mm isotropic voxels at the origin. Extraction is
        deferred until to_anatomy_collection(), just as for file-based construction.
        """
        instance = cls.__new__(cls)
        instance.source_path = None
        instance._initialize(
            masks_zyx,
            normalize_labelmap(labelmap),
            np.diag([0.001, 0.001, 0.001, 1.0])
            if affine_xyz_to_imaging_m is None
            else affine_xyz_to_imaging_m,
        )
        return instance

    def _initialize(self, volume, labelmap, affine):
        """Validate and copy common input state for both file and array constructors."""
        volume, affine = np.asarray(volume), np.asarray(affine, dtype=float)
        if (
            volume.ndim != 3
            or min(volume.shape) == 0
            or not np.isfinite(volume).all()
            or np.any(volume < 0)
            or np.any(volume != np.floor(volume))
            or np.any(volume > np.iinfo(np.int32).max)
        ):
            raise ValueError("Expected a finite non-negative 3D integer segmentation")
        if (
            affine.shape != (4, 4)
            or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(affine[:3, :3])) < 1e-18
        ):
            raise ValueError("Expected an invertible finite voxel-to-imaging affine")
        present = set(map(int, np.unique(volume))) - {0}
        missing = present - labelmap.keys()
        if missing:
            raise ValueError(f"Present IDs missing from labelmap: {sorted(missing)}")
        self.masks_zyx = volume.astype(np.int32, copy=True)
        self.labelmap = dict(labelmap)
        self.affine_xyz_to_imaging_m = affine.copy()
        self._names = {}
        for label_id, raw in labelmap.items():
            name = canonical_name(raw)
            if label_id == 0 or SKIP_STRUCTURES.fullmatch(name):
                continue
            if not name:
                raise ValueError(f"Empty canonical name for {raw!r}")
            self._names[label_id] = name

    def to_anatomy_collection(
        self,
        *,
        strict: bool = False,
        configuration=None,
    ) -> AnatomyCollection:
        """Extract present labels with the bundled imaging_to_mesh.mask_to_mesh.

        Vertices are centered per structure in physical XYZ meters, with a
        proper rigid local_to_body transform. Orientation and voxel scaling
        are baked into the vertices, so even sheared images have rigid frames.
        The shared body origin is the extracted anatomy's bounding-box center.
        The collection retains source coordinates and scan bounds as import metadata.
        Optional configuration is a YAML path or AnatomyConfiguration policy.
        Disabled meshes are retained for later re-enabling, not skipped at import.
        """
        from ..imaging_to_mesh import mask_to_mesh
        from ._labels import anatomy_from_labels

        body = anatomy_from_labels(self._names, strict=strict)
        bounds = []
        for name, structure in body.structures.items():
            ids = [label_id for label_id, value in self._names.items() if value == name]
            mask = np.isin(self.masks_zyx, ids)
            indices = np.argwhere(mask)
            if not len(indices):
                continue
            low, high = indices.min(0), indices.max(0) + 1
            window = mask[tuple(slice(a, b) for a, b in zip(low, high))]
            # Unit-spacing converter output is voxel XYZ; apply the entire
            # original affine afterwards (not only spacing and origin).
            vertices, faces = mask_to_mesh(window)
            vertices = transform_points(
                vertices + low[::-1], self.affine_xyz_to_imaging_m
            )
            # The converter returns outward XYZ winding. A reflected image
            # affine must reverse it to keep signed volume/normals outward.
            if np.linalg.det(self.affine_xyz_to_imaging_m[:3, :3]) < 0:
                faces = faces[:, ::-1].copy()
            center = vertices.mean(0)
            structure.vertices = vertices - center
            structure.faces = faces
            structure.local_to_body[:3, 3] = center
            bounds.extend([vertices.min(0), vertices.max(0)])
        origin = (
            (np.min(bounds, axis=0) + np.max(bounds, axis=0)) / 2
            if bounds
            else np.zeros(3)
        )
        body_to_imaging = np.eye(4)
        body_to_imaging[:3, 3] = origin
        body.body_to_imaging = body_to_imaging
        for structure in body.structures.values():
            if structure.mesh.vertices is not None:
                structure.local_to_body[:3, 3] -= origin
                structure.local_to_world = structure.local_to_body.copy()
        body.body_to_voxel = (
            np.linalg.inv(self.affine_xyz_to_imaging_m) @ body_to_imaging
        )
        body.source_shape_xyz = tuple(self.masks_zyx.shape[::-1])
        body.source_path = (
            str(self.source_path) if self.source_path is not None else None
        )
        if configuration is not None:
            body.configure(configuration)
        return body
