# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segment the bundled CT with NV-Segment-CTMR and view its HumanBody."""

from pathlib import Path

from _importer_viewer import parser, show
from patient_digital_twin.importers import NVSegmentImporter


def main():
    """Accept a CT/MR NIfTI and optional bundle Python environment."""
    cli = parser(__doc__)
    cli.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).parent / "data/nv_ct_high_resolution/ct.nii.gz",
    )
    cli.add_argument("--modality", choices=("CT", "MR"), default="CT")
    cli.add_argument("--bundle-root", help="Inner NV-Segment-CTMR bundle directory")
    cli.add_argument("--python", dest="python_executable")
    args = cli.parse_args()
    importer = NVSegmentImporter(
        args.image,
        bundle_root=args.bundle_root,
        modality=args.modality,
        python_executable=args.python_executable,
    )
    report = show(importer, args)
    if args.validate_only and not report["all_inside"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
