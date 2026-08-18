# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for :mod:`imaging_to_mesh`."""

from __future__ import annotations

import argparse

from .converter import convert_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert NIfTI/NRRD segmentation labelmaps to OBJ and OpenUSD.")
    parser.add_argument("input_path", help="Input labelmap file or directory")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "-p",
        "--pattern",
        default=r".*label\.(nii(\.gz)?|nrrd)$",
        help="Regex used when input_path is a directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = convert_path(args.input_path, args.output_dir, pattern=args.pattern)
    for result in results:
        print(result.usd_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
