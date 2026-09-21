# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Segment the bundled CT with TotalSegmentator's Python API and open the viewer."""

from pathlib import Path

from _importer_viewer import parser, show
from patient_digital_twin.importers import TotalSegmentatorImporter


def main():
    """Accept CT or MR input; the segmentation dependency is optional."""
    cli = parser(__doc__)
    cli.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).parent / "data/nv_ct_high_resolution/ct.nii.gz",
    )
    cli.add_argument("--modality", choices=("CT", "MR"), default="CT")
    cli.add_argument("--device", default="gpu")
    args = cli.parse_args()
    report = show(
        TotalSegmentatorImporter(
            args.image, modality=args.modality, device=args.device
        ),
        args,
    )
    if args.validate_only and not report["all_inside"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
