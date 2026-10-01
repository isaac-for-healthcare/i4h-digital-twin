# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate legacy CT artifacts independently from a CT and optional vessel mask."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .artifacts import DEFAULT_PRESET, PRESETS, write_artifacts
from .ct.dicom_ingest import load_nifti_hu


def main(argv=None):
    """Require matching physical grids and atomically publish a new artifact folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--vessel-mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hu-to-mu", choices=sorted(PRESETS), default=DEFAULT_PRESET)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    ct = load_nifti_hu(args.ct)
    mask = None
    if args.vessel_mask:
        labels = load_nifti_hu(args.vessel_mask)
        if (
            labels.hu_zyx.shape != ct.hu_zyx.shape
            or not np.allclose(labels.spacing_zyx_mm, ct.spacing_zyx_mm)
            or not np.allclose(labels.origin_xyz_mm, ct.origin_xyz_mm)
            or not np.allclose(labels.direction, ct.direction)
        ):
            raise ValueError("Vessel mask must match the CT physical grid")
        mask = labels.hu_zyx > 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".ct-artifacts-") as temp:
        folder = Path(temp) / "artifacts"
        write_artifacts(
            ct, folder, source=args.ct, vessel_mask=mask, hu_to_mu_preset=args.hu_to_mu
        )
        shutil.move(str(folder), output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
