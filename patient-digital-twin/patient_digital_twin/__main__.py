# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line pipeline: run NV-Segment or NV-Generate and export a patient digital twin.

``python -m patient_digital_twin --source nvsegment --input ct.nii.gz --classes aorta
--bundle-root <bundle> --format bundle --output out/`` segments a CT (NIfTI, DICOM
directory, or ``volume.yaml``), meshes only the requested classes, extracts missing
vessel centerlines, attaches the CT, and writes a USD file or a schema-2 bundle.
``--source nvgenerate`` synthesizes the CT instead. Run with ``--help`` for all options.

Main functions:

- ``run_pipeline``: the whole pipeline as a keyword-only Python call (used by
  i4h-workflows); raises ``ValueError``/``FileExistsError`` before inference for bad options.
- ``parser`` / ``main``: the argparse front end.
- ``class_names``: parse and validate ``--classes`` values into catalog names.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import nibabel as nib
import numpy as np

from .anatomy import CATALOG, canonical_name, is_vessel
from .body import HumanBody
from .export import USD_SUFFIXES
from .importers import NVGenerateImporter, NVSegmentImporter
from .scan_volume import ScanVolume


def class_names(values: str | Iterable[str]) -> tuple[str, ...]:
    """Parse class names separated by spaces or commas into unique canonical catalog names.

    Example:
        ``class_names(["aorta,liver", "Left Kidney"]) == ("aorta", "liver", "kidney_left")``
    """
    values = [values] if isinstance(values, str) else values
    names = tuple(dict.fromkeys(canonical_name(p.strip()) for v in values for p in v.split(",") if p.strip()))
    if not names:
        raise ValueError("Provide at least one anatomical class")
    if unknown := set(names) - CATALOG.keys():
        raise ValueError(f"Unknown anatomical classes: {sorted(unknown)}")
    return names


def _load_scan(input: Path, series_uid: str | None) -> ScanVolume:
    """Read a DICOM directory (LPS mm), ``volume.yaml`` artifact, or NIfTI as a native ScanVolume."""
    from . import scan_volume

    if input.is_dir():
        conversion = scan_volume.Conversion(world_frame="LPS", world_unit="mm")
        return scan_volume.from_dicom(input, series_uid=series_uid, conversion=conversion)
    if input.suffix in {".yaml", ".yml"}:
        return scan_volume.load_artifact(input)
    if input.is_file() and input.name.lower().endswith((".nii", ".nii.gz")):
        return scan_volume.from_nifti(input)
    raise ValueError("input must be a DICOM directory, .nii/.nii.gz image, or volume.yaml")


def run_pipeline(
    *,
    source: str,
    classes: str | Iterable[str],
    output: str | Path,
    input: str | Path | None = None,
    format: str = "usd",
    modality: str = "CT",
    bundle_root: str | Path | None = None,
    source_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    centerline_spacing_mm: float = 1.5,
    series_uid: str | None = None,
) -> Path:
    """Run NV-Segment or NV-Generate and export only the requested anatomy.

    Args:
        source: ``"nvsegment"`` (requires ``input``) or ``"nvgenerate"`` (no ``input``).
        classes: Catalog class names; every one must get a mesh or ``ValueError`` is raised.
        output: New ``.usd/.usda/.usdc`` file for ``format="usd"``, or new directory for ``"bundle"``.
        input: CT/MR NIfTI, DICOM CT directory, or ``volume.yaml``.
        format: ``"usd"`` or ``"bundle"`` (bundle needs CT and at least one vessel class).
        modality: ``"CT"`` or ``"MR"`` (MR is USD-only, without embedded imaging).
        bundle_root / source_root / python_executable: backend checkout and optional interpreter.
        centerline_spacing_mm: Voxel size for vessel centerline extraction.
        series_uid: DICOM series to use when a directory contains several.

    Returns:
        The written USD path, or the bundle's ``patient_twin.yaml``.

    Example:
        ``run_pipeline(source="nvsegment", input="ct.nii.gz", classes=["aorta"], bundle_root=root,
        format="bundle", output="out")``
    """
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
    scan: ScanVolume | None
    source_path = None
    if source == "nvsegment":
        input_path = Path(str(input)).expanduser().resolve()
        scan = _load_scan(input_path, series_uid)
        image = nib.Nifti1Image(scan.values_kji.transpose(2, 1, 0), scan.ijk_to_ras_m)
        image.header.set_xyzt_units("meter")
        segmenter = NVSegmentImporter(image, bundle_root=bundle_root, modality=modality,
                                      python_executable=python_executable)
        anatomy = segmenter.to_anatomy_collection(names=names)
        scan, source_path = (scan, str(input_path)) if modality == "CT" else (None, None)
    else:
        generator = NVGenerateImporter(source_root=source_root, python_executable=python_executable)
        anatomy = generator.to_anatomy_collection(names=names)
        scan = generator.ct_scan
    if absent := [n for n in names if n not in anatomy.structures or anatomy.structures[n].is_empty]:
        raise ValueError(f"Requested classes have no mesh in the model output: {absent}")
    body = HumanBody(anatomy)
    if scan is not None:
        body.attach_scan(scan, source_path=source_path)
    if missing := [n for n in vessel_names if body.anatomy.structures[n].centerline is None]:
        body.extract_topology(names=missing, spacing_m=centerline_spacing_mm * 0.001)
    if format == "bundle":
        return body.export_patient_twin(output, vessel_names=vessel_names, ct_exterior=True)
    return body.export_to_usd(output)


def parser() -> argparse.ArgumentParser:
    """Build the CLI parser; ``vars(parser().parse_args())`` matches ``run_pipeline``'s keywords."""
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: run the pipeline and print the output path; option errors exit with status 2."""
    cli = parser()
    try:
        result = run_pipeline(**vars(cli.parse_args(argv)))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        cli.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
