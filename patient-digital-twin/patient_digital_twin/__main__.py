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
from .importers._segmentation import canonical_name
from .structures import Kind


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
    centerline_spacing_mm=1.5,
    series_uid=None,
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
    if format not in {"usd", "bundle"}:
        raise ValueError("format must be usd or bundle")
    if not np.isfinite(centerline_spacing_mm) or centerline_spacing_mm <= 0:
        raise ValueError("centerline_spacing_mm must be finite and positive")
    if format == "bundle" and modality != "CT":
        raise ValueError("bundle output requires CT in Hounsfield units")
    if format == "bundle" and not any(
        CATALOG[name] == Kind.VESSEL or name == "portal_vein_and_splenic_vein"
        for name in names
    ):
        raise ValueError("bundle output requires at least one vessel class")
    output = Path(output).expanduser().resolve()
    if format == "usd" and output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("USD output must end with .usd, .usda or .usdc")
    if output.exists():
        raise FileExistsError(f"Use a new output path: {output}")
    if source == "nvsegment":
        if input is None:
            raise ValueError(
                "nvsegment requires --input pointing to a 3D medical NIfTI image, DICOM CT directory, or volume.yaml"
            )
        input = Path(input).expanduser().resolve()
        from . import scan_volume

        if input.is_dir():
            scan = scan_volume.from_dicom(
                input,
                series_uid=series_uid,
                conversion=scan_volume.Conversion(world_frame="LPS", world_unit="mm"),
            )
            image = image_input(scan.values_kji.transpose(2, 1, 0), scan.ijk_to_ras_m)
        elif input.suffix in {".yaml", ".yml"}:
            scan = scan_volume.load_artifact(input)
            image = image_input(scan.values_kji.transpose(2, 1, 0), scan.ijk_to_ras_m)
        elif input.is_file() and str(input).lower().endswith((".nii", ".nii.gz")):
            scan = scan_volume.from_nifti(input)
            image = image_input(input)
        else:
            raise ValueError(
                "input must be a DICOM directory, .nii/.nii.gz image, or volume.yaml"
            )
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
    if source == "nvsegment" and modality == "CT":
        body.AttachScan(scan, source_path=str(input))
    elif source == "nvgenerate":
        if getattr(importer, "ct_scan", None) is not None:
            body.AttachScan(importer.ct_scan)
        else:
            body.AttachImaging(
                importer.ct_volume_zyx, voxel_to_imaging=importer.ct_voxel_to_imaging
            )
    vessel_names = tuple(
        name
        for name in names
        if body.anatomy.structures[name].kind == Kind.VESSEL
        or name == "portal_vein_and_splenic_vein"
    )
    if format == "bundle" and not vessel_names:
        raise ValueError("bundle output requires at least one vessel class")
    missing = [
        name
        for name in vessel_names
        if body.anatomy.structures[name].centerline is None
    ]
    if missing:
        body.extract_topology(names=missing, spacing_m=centerline_spacing_mm * 0.001)
    if format == "bundle":
        return body.export_patient_twin(
            output,
            vessel_names=vessel_names,
            exterior="ct",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    return body.export_to_usd(output)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", required=True, choices=("nvgenerate", "nvsegment"))
    result.add_argument(
        "--input",
        type=Path,
        help="CT/MR NIfTI, DICOM CT directory, or volume.yaml; required for nvsegment",
    )
    result.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Named anatomy classes (spaces or commas)",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--series-uid", help="DICOM series to select when a directory contains several"
    )
    result.add_argument("--format", choices=("usd", "bundle"), default="usd")
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
    result.add_argument("--centerline-spacing-mm", type=float, default=1.5)
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
