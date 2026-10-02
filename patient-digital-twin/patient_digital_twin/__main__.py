# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate or segment named anatomy and export a patient digital twin."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from .body import CATALOG, HumanBody, canonical_name, is_vessel
from .export import USD_SUFFIXES
from .importers import NVGenerateImporter, NVSegmentImporter


def class_names(values):
    """Canonical catalog names from strings separated by spaces or commas."""
    values = [values] if isinstance(values, str) else values
    names = tuple(dict.fromkeys(canonical_name(p.strip()) for v in values for p in v.split(",") if p.strip()))
    if not names:
        raise ValueError("Provide at least one anatomical class")
    if unknown := set(names) - CATALOG.keys():
        raise ValueError(f"Unknown anatomical classes: {sorted(unknown)}")
    return names


def _load_scan(input, series_uid):
    from . import scan_volume

    if input.is_dir():
        conversion = scan_volume.Conversion(world_frame="LPS", world_unit="mm")
        return scan_volume.from_dicom(input, series_uid=series_uid, conversion=conversion)
    if input.suffix in {".yaml", ".yml"}:
        return scan_volume.load_artifact(input)
    if input.is_file() and input.name.lower().endswith((".nii", ".nii.gz")):
        return scan_volume.from_nifti(input)
    raise ValueError("input must be a DICOM directory, .nii/.nii.gz image, or volume.yaml")


def run_pipeline(*, source, classes, output, input=None, format="usd", modality="CT", bundle_root=None,
                 source_root=None, python_executable=None, centerline_spacing_mm=1.5, series_uid=None):
    """Run NV-Segment or NV-Generate and export only the requested anatomy as USD or a bundle."""
    names, modality = class_names(classes), modality.upper()
    vessel_names = tuple(n for n in names if is_vessel(n))
    output = Path(output).expanduser().resolve()
    for failed, message in (
        (source not in {"nvgenerate", "nvsegment"}, "source must be nvgenerate or nvsegment"),
        (modality not in {"CT", "MR"}, "modality must be CT or MR"),
        (format not in {"usd", "bundle"}, "format must be usd or bundle"),
        (not np.isfinite(centerline_spacing_mm) or centerline_spacing_mm <= 0,
         "centerline_spacing_mm must be finite and positive"),
        (format == "bundle" and modality != "CT", "bundle output requires CT in Hounsfield units"),
        (format == "bundle" and not vessel_names, "bundle output requires at least one vessel class"),
        (format == "usd" and output.suffix.lower() not in USD_SUFFIXES,
         "USD output must end with .usd, .usda or .usdc"),
        (source == "nvsegment" and input is None,
         "nvsegment requires --input pointing to a 3D NIfTI image, DICOM CT directory, or volume.yaml"),
        (source == "nvgenerate" and input is not None, "nvgenerate creates its own CT; omit --input"),
        (source == "nvgenerate" and modality != "CT", "nvgenerate currently generates CT only"),
    ):
        if failed:
            raise ValueError(message)
    if output.exists():
        raise FileExistsError(f"Use a new output path: {output}")
    if source == "nvsegment":
        input = Path(input).expanduser().resolve()
        scan = _load_scan(input, series_uid)
        image = nib.Nifti1Image(scan.values_kji.transpose(2, 1, 0), scan.ijk_to_ras_m)
        image.header.set_xyzt_units("meter")
        importer = NVSegmentImporter(image, bundle_root=bundle_root, modality=modality,
                                     python_executable=python_executable)
    else:
        importer = NVGenerateImporter(source_root=source_root, python_executable=python_executable)
    anatomy = importer.to_anatomy_collection(names=names)
    if absent := [n for n in names if n not in anatomy.structures or anatomy.structures[n].is_empty]:
        raise ValueError(f"Requested classes have no mesh in the model output: {absent}")
    body = HumanBody(anatomy)
    if source == "nvgenerate":
        body.attach_scan(importer.ct_scan)
    elif modality == "CT":
        body.attach_scan(scan, source_path=str(input))
    if missing := [n for n in vessel_names if body.anatomy.structures[n].centerline is None]:
        body.extract_topology(names=missing, spacing_m=centerline_spacing_mm * 0.001)
    if format == "bundle":
        return body.export_patient_twin(output, vessel_names=vessel_names, ct_exterior=True)
    return body.export_to_usd(output)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", required=True, choices=("nvgenerate", "nvsegment"))
    result.add_argument("--input", type=Path, help="CT/MR NIfTI, DICOM CT directory, or volume.yaml (nvsegment)")
    result.add_argument("--classes", nargs="+", required=True, help="Named anatomy classes (spaces or commas)")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--series-uid", help="DICOM series to select when a directory contains several")
    result.add_argument("--format", choices=("usd", "bundle"), default="usd")
    result.add_argument("--modality", choices=("CT", "MR"), default="CT")
    result.add_argument("--bundle-root", type=Path, help="Inner NV-Segment-CTMR bundle directory")
    result.add_argument("--source-root", type=Path, help="NV-Generate-CTMR checkout")
    result.add_argument("--python", dest="python_executable", help="Python interpreter for model inference")
    result.add_argument("--centerline-spacing-mm", type=float, default=1.5)
    return result


def main(argv=None):
    cli = parser()
    try:
        result = run_pipeline(**vars(cli.parse_args(argv)))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        cli.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
