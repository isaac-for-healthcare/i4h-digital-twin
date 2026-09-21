# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the bundled colon STL using its sample-derived default body placement."""

from pathlib import Path

from _importer_viewer import parser, show
from patient_digital_twin.importers import SimpleImporter


def main():
    """Run: python examples/simple_importer.py [--validate-only]."""
    cli = parser(__doc__)
    cli.set_defaults(
        parameters=Path(__file__).parent / "data/colon_soma_parameters.json"
    )
    args = cli.parse_args()
    importer = SimpleImporter({"colon": Path(__file__).parent / "data/colon.stl"})
    report = show(importer, args)
    if args.validate_only and not report["all_inside"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
