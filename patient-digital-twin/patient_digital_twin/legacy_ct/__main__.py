# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate legacy CT artifacts independently from a CT and optional vessel mask."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ..exporters.native import native_centerline
from ..scan_volume import from_nifti


def main(argv=None):
    """Require matching physical grids and atomically publish a new artifact folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--vessel-mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    ct = from_nifti(args.ct)
    mask = None
    if args.vessel_mask:
        labels = from_nifti(args.vessel_mask)
        if labels.values.shape != ct.values.shape or not np.allclose(
            labels.ijk_to_ras_m, ct.ijk_to_ras_m
        ):
            raise ValueError("Vessel mask must match the CT physical grid")
        if not np.isin(labels.values, [0, 1]).all():
            raise ValueError("Vessel mask must be binary")
        mask = labels.values > 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".ct-artifacts-") as temp:
        folder = Path(temp) / "artifacts"
        ct.save(folder)
        if mask is not None:
            np.save(folder / "vessel_mask.npy", mask.astype(np.uint8))
            for name, values in zip(
                ("centerline_points", "centerline_edges", "centerline_radii"),
                native_centerline(mask, ct),
            ):
                np.save(folder / f"{name}.npy", values)
        shutil.move(str(folder), output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
