# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared input coordinates, optional-runtime checks, and catalog coverage."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

from ..catalog import CATALOG
from ._segmentation import SegmentationImporter, canonical_name


def image_input(image, affine_xyz_to_imaging_m=None):
    """Accept NIfTI/path or a XYZ NumPy volume with an explicit meter affine.

    Arrays cannot encode spacing/orientation; require their affine instead of
    silently assuming physical dimensions. File/header units are respected.
    """
    if isinstance(image, (str, Path)):
        image = nib.load(str(image))
    elif isinstance(image, np.ndarray):
        if affine_xyz_to_imaging_m is None:
            raise ValueError(
                "NumPy images require affine_xyz_to_imaging_m (XYZ indices to meters)"
            )
        image = nib.Nifti1Image(
            np.asarray(image, dtype=np.float32), affine_xyz_to_imaging_m
        )
        image.header.set_xyzt_units("meter")
    if not isinstance(image, nib.spatialimages.SpatialImage):
        raise TypeError("Expected a NIfTI image/path or an XYZ NumPy array")
    affine_m = SegmentationImporter._affine_m(image)
    if not np.isfinite(affine_m).all() or abs(np.linalg.det(affine_m[:3, :3])) < 1e-18:
        raise ValueError("Image affine must be finite and invertible")
    # Backends generally assume NIfTI millimeters regardless of header units.
    affine_mm = affine_m.copy()
    affine_mm[:3] *= 1000
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError("Input image must be a 3D volume")
    if not np.isfinite(data).all():
        raise ValueError("Image contains non-finite intensities")
    result = nib.Nifti1Image(data.astype(np.float32), affine_mm)
    result.header.set_xyzt_units("mm")
    return result


def catalog_labels(labelmap):
    """Keep supported canonical names; never infer label IDs from voxel values."""
    return {
        int(i): canonical_name(name)
        for i, name in labelmap.items()
        if canonical_name(name) in CATALOG
    }


def segmentation_body(image, labelmap, *, configuration=None):
    """Mesh catalog labels and keep unsupported/unobserved catalog entries empty."""
    labels = catalog_labels(labelmap)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.isfinite(data).all() or np.any(data != np.floor(data)):
        raise ValueError(
            "Backend output must be a finite integer-valued 3D segmentation"
        )
    data = np.where(np.isin(data, list(labels)), data, 0)
    # Include every catalog entry, without reassigning a backend's real IDs.
    mapping = dict(labels)
    next_id = max(mapping, default=0) + 1
    for name in CATALOG:
        if name not in mapping.values():
            mapping[next_id] = name
            next_id += 1
    importer = SegmentationImporter.from_array(
        data.transpose(2, 1, 0),
        mapping,
        affine_xyz_to_imaging_m=SegmentationImporter._affine_m(image),
    )
    return importer.to_human_body(configuration=configuration)


def coverage(body, supported, *, backend, **details):
    """Report actual geometry separately from unsupported and absent labels."""
    present = {n for n, s in body.structures.items() if s.mesh.vertices is not None}
    supported = set(supported)
    result = {
        "backend": backend,
        "present": sorted(present),
        "unsupported": sorted(set(CATALOG) - supported),
        "absent": sorted(supported - present),
        **details,
    }
    if result["unsupported"] or result["absent"]:
        warnings.warn(
            f"{backend}: {len(present)}/{len(CATALOG)} catalog structures have meshes; "
            f"unsupported={result['unsupported']}; absent={result['absent']}",
            stacklevel=2,
        )
    return result


def runtime(python_executable, modules, install):
    """Fail before inference with a pip-install hint in the chosen interpreter."""
    python = str(python_executable or sys.executable)
    check = subprocess.run(
        [
            python,
            "-c",
            (
                "import importlib.util, json, sys; "
                "print(json.dumps([m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]))"
            ),
            *modules,
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    missing = json.loads(check.stdout)
    if missing:
        raise ImportError(
            f"Missing optional dependencies: {', '.join(missing)}. Run: {python} -m pip install {install}"
        )
    return python


def run_backend(command, *, cwd=None):
    """Run a backend with visible progress; fail rather than import stale output."""
    subprocess.run([str(item) for item in command], cwd=cwd, check=True)
