# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write native CT/mask/centerline arrays; all centerline calculation is in topology."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .scan_volume import from_nifti
from .topology import native_centerline


def write_centerline(output, points, edges, radii):
    """Write scan-unit XYZ points/radii and point-index edges, without conversion."""
    points, edges, radii = map(np.asarray, (points, edges, radii))
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2
            or radii.shape != (len(points),) or not np.isfinite(points).all()
            or not np.isfinite(radii).all() or np.any(radii < 0)
            or edges.ndim != 2 or edges.shape[1:] != (2,) or not len(edges)
            or not np.issubdtype(edges.dtype, np.integer)
            or edges.min() < 0 or edges.max() >= len(points)):
        raise ValueError("Invalid physical centerline graph")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, values in zip(("centerline_points", "centerline_edges", "centerline_radii"), (points, edges, radii)):
        artifacts[name] = f"{name}.npy"
        np.save(output / artifacts[name], values, allow_pickle=False)
    return artifacts


def write_artifacts(scan, output, *, vessel_mask=None):
    """Atomically export native HU/YAML and optional scan-grid vessel topology."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    graph = None
    if vessel_mask is not None:
        mask = np.asarray(vessel_mask)
        if mask.shape != scan.values.shape:
            raise ValueError("Vessel mask must match the scan grid")
        graph = native_centerline(mask, scan)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".ct-artifacts-") as temp:
        folder = Path(temp) / "artifacts"
        scan.save(folder)
        if graph is not None:
            np.save(folder / "vessel_mask.npy", mask.astype(np.uint8))
            write_centerline(folder, *graph)
        shutil.move(str(folder), output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--vessel-mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    scan = from_nifti(args.ct)
    mask = None
    if args.vessel_mask:
        labels = from_nifti(args.vessel_mask)
        if labels.values.shape != scan.values.shape or not np.allclose(labels.ijk_to_ras_m, scan.ijk_to_ras_m):
            raise ValueError("Vessel mask must match the CT physical grid")
        mask = labels.values
    print(write_artifacts(scan, args.output, vessel_mask=mask))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
