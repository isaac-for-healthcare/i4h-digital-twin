# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native scan geometry shared by the bundle and USD exporters."""

import numpy as np

from ..geometry import transform_points
from ..scan_volume import from_array


def scan_for_body(body):
    imaging = body.imaging
    if imaging is None:
        return None
    if imaging.source_scan is not None:
        return imaging.source_scan
    reverse = np.eye(4)
    reverse[:3, :3] = np.eye(3)[:, [2, 1, 0]]
    return from_array(
        imaging.volume,
        imaging.voxel_to_imaging @ reverse,
        array_axes="kji",
        world_unit="m",
    )


def scan_from_body_m(body, scan):
    registration = (
        body.imaging.body_to_imaging
        if body.imaging is not None
        else body.anatomy.body_to_imaging
    )
    if registration is None:
        return np.eye(4)
    ras_to_scan = (
        np.diag([-1.0, -1.0, 1.0, 1.0])
        if scan is not None and scan.frame == "LPS"
        else np.eye(4)
    )
    return ras_to_scan @ registration


def mask_on_scan(body, scan, names):
    anatomy = body.anatomy
    labels = anatomy.source_segmentation
    if labels is not None:
        if labels.shape != scan.values_kji.shape or not np.allclose(
            anatomy.source_voxel_to_ras_m, scan.ijk_to_ras_m, atol=1e-9, rtol=1e-6
        ):
            raise ValueError(
                "Segmentation and attached scan must share the same physical grid"
            )
        ids = [i for i, name in anatomy.source_label_names.items() if name in names]
        mask = np.isin(labels, ids)
    else:
        # Mesh-only inputs have no source labels. Rasterize in native voxel indices.
        from ..topology import voxelize_mesh

        mask = np.zeros(scan.values_kji.shape, bool)
        for name in names:
            structure = anatomy.structures[name]
            matrix = (
                np.linalg.inv(scan.ijk_to_ras_m)
                @ body.imaging.body_to_imaging
                @ structure.local_to_body
            )
            vertices = transform_points(structure.mesh.vertices, matrix)
            mask |= voxelize_mesh(
                vertices,
                structure.mesh.faces,
                shape_zyx=mask.shape,
                spacing_zyx_m=(1.0, 1.0, 1.0),
                origin_xyz_m=(0.0, 0.0, 0.0),
            )
    if not mask.any():
        raise ValueError("Selected vessels have no foreground in the scan")
    return mask.transpose(["kji".index(c) for c in scan.array_axes])
