# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate or segment named anatomy and export a patient digital twin."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .catalog import CATALOG
from .human import HumanBody
from .importers import NVGenerateImporter, NVSegmentImporter
from .importers._common import image_input
from .importers._segmentation import SegmentationImporter, canonical_name


def class_names(values):
    """Accept canonical names separated by spaces or commas."""
    if isinstance(values, str):
        values = [values]
    names = tuple(
        dict.fromkeys(
            canonical_name(part.strip())
            for value in values
            for part in value.split(",")
            if part.strip()
        )
    )
    if not names:
        raise ValueError("Provide at least one anatomical class")
    unknown = set(names) - CATALOG.keys()
    if unknown:
        raise ValueError(f"Unknown anatomical classes: {sorted(unknown)}")
    return names


def run_pipeline(
    *,
    source,
    classes,
    output,
    input=None,
    format="usd",
    modality="CT",
    bundle_root=None,
    source_root=None,
    python_executable=None,
):
    """Run an optional model backend and export only the requested meshes.

    NV-Segment accepts a 3D NIfTI image (.nii or .nii.gz). Backends may use a
    separate Python environment; the library/export dependencies stay here.
    Output must be new. Missing requested anatomy is an error.
    """
    if source not in {"nvgenerate", "nvsegment"}:
        raise ValueError("source must be nvgenerate or nvsegment")
    names = class_names(classes)
    modality = modality.upper()
    if modality not in {"CT", "MR"}:
        raise ValueError("modality must be CT or MR")
    if format != "usd":
        raise ValueError("format must be usd")
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("USD output must end with .usd, .usda or .usdc")
    if output.exists():
        raise FileExistsError(f"Use a new output path: {output}")
    if source == "nvsegment":
        if input is None:
            raise ValueError(
                "nvsegment requires --input pointing to a 3D medical NIfTI image"
            )
        input = Path(input).expanduser().resolve()
        if not input.is_file() or not str(input).lower().endswith((".nii", ".nii.gz")):
            raise ValueError("input must be an existing .nii or .nii.gz image")
        image = image_input(input)
        importer = NVSegmentImporter(
            image,
            bundle_root=bundle_root,
            modality=modality,
            python_executable=python_executable,
        )
    else:
        if input is not None:
            raise ValueError("nvgenerate creates its own CT; omit --input")
        if modality != "CT":
            raise ValueError("nvgenerate currently generates CT only")
        importer = NVGenerateImporter(
            source_root=source_root, python_executable=python_executable
        )
    anatomy = importer.to_anatomy_collection(names=names)
    absent = [
        name
        for name in names
        if name not in anatomy.structures or anatomy.structures[name].is_empty
    ]
    if absent:
        raise ValueError(
            f"Requested classes have no mesh in the model output: {absent}"
        )
    body = HumanBody(anatomy)
    if source == "nvsegment":
        body.AttachImaging(
            np.asarray(image.dataobj).transpose(2, 1, 0),
            voxel_to_imaging=SegmentationImporter._affine_m(image),
            source_path=str(input),
            modality=modality,
        )
    else:
        body.AttachImaging(
            importer.ct_volume_zyx, voxel_to_imaging=importer.ct_voxel_to_imaging
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    return body.export_to_usd(output)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", required=True, choices=("nvgenerate", "nvsegment"))
    result.add_argument(
        "--input", type=Path, help="3D CT/MR NIfTI; required for nvsegment"
    )
    result.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Named anatomy classes (spaces or commas)",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--format", choices=("usd",), default="usd")
    result.add_argument("--modality", choices=("CT", "MR"), default="CT")
    result.add_argument(
        "--bundle-root", type=Path, help="Inner NV-Segment-CTMR bundle directory"
    )
    result.add_argument("--source-root", type=Path, help="NV-Generate-CTMR checkout")
    result.add_argument(
        "--python",
        dest="python_executable",
        help="Python interpreter for model inference",
    )
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        result = run_pipeline(**vars(args))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        cli.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
