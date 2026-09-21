# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a fresh mask with NV-Generate-CTMR and view its HumanBody."""

from _importer_viewer import parser, show
from patient_digital_twin.importers import NVGenerateImporter


def main():
    """Use the optional upstream checkout and its pip-installed environment."""
    cli = parser(__doc__)
    cli.set_defaults(partial_preview=True)
    cli.add_argument(
        "--source-root", help="NV-Generate-CTMR checkout; defaults to NV_GENERATE_ROOT"
    )
    cli.add_argument(
        "--python",
        dest="python_executable",
        help="Python with upstream requirements installed",
    )
    args = cli.parse_args()
    importer = NVGenerateImporter(
        source_root=args.source_root, python_executable=args.python_executable
    )
    report = show(importer, args)
    if args.validate_only and not report["all_inside"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
