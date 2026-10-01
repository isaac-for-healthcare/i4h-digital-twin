# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated compatibility writer for the navigation volume and graph artifacts."""

import json
from pathlib import Path

import numpy as np

from .centerline import centerline_from_mask


def write_artifacts(
    ct,
    output,
    *,
    source="numpy",
    vessel_mask=None,
    centerline=None,
):
    """Write HU, spatial metadata, and optional mask/graph in LPS millimeters.

    ``ct`` supplies HU in ZYX order plus spacing, origin, and direction. The
    optional ``centerline`` is (points_mm, edges, radii_mm); a missing graph is
    calculated from the supplied mask. No segmentation model is run here.
    """
    hu = np.asarray(ct.hu_zyx)
    spacing = np.asarray(ct.spacing_zyx_mm, float)
    origin = np.asarray(ct.origin_xyz_mm, float)
    if hu.ndim != 3 or not hu.size or not np.isfinite(hu).all():
        raise ValueError("Expected a non-empty, finite 3D CT volume")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Expected positive finite ZYX spacing")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Expected a finite XYZ origin")
    if ct.anatomical_frame != "LPS" or not np.allclose(
        np.asarray(ct.direction).reshape(3, 3), np.eye(3), atol=1e-4
    ):
        raise ValueError(
            "Navigation artifacts require axis-aligned LPS CT; resample oblique inputs first"
        )
    if centerline is not None and vessel_mask is None:
        raise ValueError("A centerline requires its vessel mask")
    if vessel_mask is not None:
        mask = np.asarray(vessel_mask)
        if (
            mask.shape != hu.shape
            or not np.isfinite(mask).all()
            or not np.isin(mask, [0, 1]).all()
            or not mask.any()
        ):
            raise ValueError(
                "Vessel mask must be nonempty, binary, and match the CT grid"
            )
        centerline = (
            centerline
            if centerline is not None
            else centerline_from_mask(mask, spacing, origin)
        )
        points, edges, radii = map(np.asarray, centerline)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) < 2
            or radii.shape != (len(points),)
            or not np.isfinite(points).all()
            or not np.isfinite(radii).all()
            or np.any(radii < 0)
            or edges.ndim != 2
            or edges.shape[1:] != (2,)
            or not len(edges)
            or not np.issubdtype(edges.dtype, np.integer)
            or edges.min() < 0
            or edges.max() >= len(points)
        ):
            raise ValueError("Invalid physical centerline graph")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "shape_zyx": list(hu.shape),
        "spacing_zyx_mm": spacing.tolist(),
        "origin_xyz_mm": origin.tolist(),
        "direction_row_major_3x3": list(ct.direction),
        "anatomical_frame": "LPS",
        "source_orientation": ct.source_orientation,
        "source": str(source),
        "array_order": "ZYX",
        "intensity_units": "HU",
        "hu_range": [float(hu.min()), float(hu.max())],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.save(output / "hu_volume.npy", hu.astype(np.float32))
    if vessel_mask is not None:
        np.save(output / "vessel_mask.npy", mask.astype(np.uint8))
        np.save(output / "centerline_points_mm.npy", points)
        np.save(output / "centerline_edges.npy", edges)
        np.save(output / "centerline_radii_mm.npy", radii)
    return output
