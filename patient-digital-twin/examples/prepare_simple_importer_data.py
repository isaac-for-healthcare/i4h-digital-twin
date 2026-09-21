# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the colon STL and importer defaults from the bundled reference scan."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from patient_digital_twin.catalog import CATALOG
from patient_digital_twin.geometry import transform_points
from patient_digital_twin.imaging_to_mesh import mask_to_mesh
from patient_digital_twin.importers._segmentation import (
    SegmentationImporter,
    canonical_name,
)
from scipy.ndimage import find_objects


def main():
    """Extract a modest-size colon surface and scan-derived anatomical placements."""
    examples = Path(__file__).parent
    sample = examples / "data/nv_ct_high_resolution"
    image = nib.load(sample / "segmentation.nii.gz")
    data = np.asanyarray(image.dataobj)
    affine = SegmentationImporter._affine_m(image)
    labelmap = {
        int(i): canonical_name(n)
        for n, i in json.loads((sample / "labels.json").read_text()).items()
        if canonical_name(n) in CATALOG
    }
    boxes = find_objects(data)
    bounds, centers = [], {}
    for i, name in labelmap.items():
        box = boxes[i - 1] if i <= len(boxes) else None
        if box is None:
            continue
        low = np.array([s.start for s in box]) - 0.5
        high = np.array([s.stop for s in box]) - 0.5
        corners = np.array(
            [
                [x, y, z]
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ]
        )
        physical = transform_points(corners, affine)
        bounds.extend([physical.min(0), physical.max(0)])
        centers[name] = transform_points((low + high) / 2, affine)
    origin = (np.min(bounds, axis=0) + np.max(bounds, axis=0)) / 2
    colon_id = next(i for i, n in labelmap.items() if n == "colon")
    box = boxes[colon_id - 1]
    # Two-voxel sampling makes a compact example while preserving physical units.
    low = np.array([s.start for s in box])
    window = (data[box][::2, ::2, ::2] == colon_id).transpose(2, 1, 0)
    vertices, faces = mask_to_mesh(window)
    vertices = transform_points(vertices * 2 + low, affine)
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]
    center = vertices.mean(0)
    centers["colon"] = center
    mesh = trimesh.Trimesh(vertices=vertices - center, faces=faces, process=False)
    mesh.export(examples / "data/colon.stl")
    # Landmarks from the reference scan are sufficient to align a subset-only body.
    importer = SegmentationImporter.from_array(
        data.transpose(2, 1, 0),
        {
            int(i): n
            for n, i in json.loads((sample / "labels.json").read_text()).items()
        },
        affine_xyz_to_imaging_m=affine,
    )
    landmarks = {
        n: (p - origin).tolist() for n, p in importer.extract_landmarks().items()
    }
    placements = {}
    for name, center in centers.items():
        matrix = np.eye(4)
        matrix[:3, 3] = center - origin
        placements[name] = matrix.tolist()
    body_to_imaging = np.eye(4)
    body_to_imaging[:3, 3] = origin
    reference = {
        "source": "examples/data/nv_ct_high_resolution/segmentation.nii.gz",
        "units": "meters",
        "body_to_imaging": body_to_imaging.tolist(),
        "landmarks": landmarks,
        "mesh_to_body": placements,
    }
    destination = examples.parent / "patient_digital_twin/importers/reference_body.json"
    destination.write_text(json.dumps(reference, indent=2) + "\n")
    print(
        f"Saved colon.stl: {len(vertices)} vertices, {len(faces)} triangles; {len(placements)} reference placements"
    )


if __name__ == "__main__":
    main()
