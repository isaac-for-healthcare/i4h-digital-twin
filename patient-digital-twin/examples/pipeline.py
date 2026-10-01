# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import anatomy, attach matching CT, extract topology, and export."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from patient_digital_twin import HumanBody, SegmentationImporter
from patient_digital_twin.catalog import CATALOG
from patient_digital_twin.importers import (
    NVGenerateImporter,
    NVSegmentImporter,
    SimpleImporter,
)

SAMPLE = Path(__file__).parent / "data/s0011"


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--source",
        choices=(
            "sample",
            "segmentation",
            "nvgenerate",
            "nvsegment",
            "simple",
        ),
        default="sample",
    )
    cli.add_argument(
        "--input", type=Path, help="Input CT, segmentation, or mesh-map JSON"
    )
    cli.add_argument("--labels", type=Path, help="Label map for segmentation input")
    cli.add_argument("--ct", type=Path, help="Matching CT for --source segmentation")
    cli.add_argument("--output", type=Path, required=True, help="New bundle directory")
    cli.add_argument("--anatomy", nargs="*", default=[], help="Optional anatomy subset")
    cli.add_argument("--source-root", type=Path, help="NVGenerate checkout")
    cli.add_argument("--bundle-root", type=Path, help="NVSegment bundle")
    cli.add_argument(
        "--python", dest="python_executable", help="NVIDIA backend interpreter"
    )
    cli.add_argument("--centerline-spacing-mm", type=float, default=1.5)
    return cli


def import_body(args):
    """Build a HumanBody and attach only the CT belonging to its anatomy."""
    source = args.source
    path = args.input.expanduser().resolve() if args.input else None
    ct = None
    if source == "sample":
        importer = SegmentationImporter(
            SAMPLE / "segmentations", names=[name.strip() for value in (args.anatomy or ["aorta"]) for name in value.split(",") if name.strip()]
        )
        ct = SAMPLE / "ct.nii.gz"
    elif source == "nvgenerate":
        importer = NVGenerateImporter(
            source_root=args.source_root, python_executable=args.python_executable
        )
    elif source == "simple":
        config = json.loads(path.read_text())
        meshes = {name: path.parent / value for name, value in config["meshes"].items()}
        importer = SimpleImporter(
            meshes,
            mesh_to_body={
                name: config.get("mesh_to_body", {}).get(name, np.eye(4))
                for name in meshes
            },
            body_to_imaging=config.get("body_to_imaging"),
        )
    elif source == "segmentation":
        importer = SegmentationImporter(path, args.labels)
        ct = args.ct
    elif source == "nvsegment":
        importer = NVSegmentImporter(
            path,
            bundle_root=args.bundle_root,
            python_executable=args.python_executable,
            modality="CT",
        )
        ct = path
    else:
        raise ValueError(f"Unsupported source: {source}")
    body = HumanBody(importer.to_anatomy_collection())
    if source == "nvgenerate":
        if getattr(importer, "ct_scan", None) is not None:
            body.AttachScan(importer.ct_scan)
        else:
            body.AttachImaging(
                importer.ct_volume_zyx, voxel_to_imaging=importer.ct_voxel_to_imaging
            )
    elif ct is not None:
        from patient_digital_twin.scan_volume import from_nifti

        body.AttachScan(from_nifti(ct), source_path=ct)
    return body


def run(args):
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    selected = {
        name.strip()
        for entry in args.anatomy
        for name in entry.split(",")
        if name.strip()
    }
    if selected - CATALOG.keys():
        raise ValueError(f"Unknown anatomy: {sorted(selected - CATALOG.keys())}")
    if not np.isfinite(args.centerline_spacing_mm) or args.centerline_spacing_mm <= 0:
        raise ValueError("Centerline spacing must be positive and finite")
    if args.source not in {"sample", "nvgenerate"} and args.input is None:
        raise ValueError(f"--source {args.source} requires --input")
    if args.source in {"sample", "nvgenerate"} and args.input is not None:
        raise ValueError(f"--source {args.source} does not accept --input")
    if args.ct is not None and args.source != "segmentation":
        raise ValueError("--ct is only supported for segmentation inputs")
    if args.source == "segmentation" and args.ct is None:
        raise ValueError("Segmentation input requires its matching --ct")
    print(f"Importing {args.source}", flush=True)
    body = import_body(args)
    requested = selected or set(body.anatomy.structures)
    available = {
        name
        for name in requested
        if name in body.anatomy.structures
        and body.anatomy.structures[name].mesh.vertices is not None
    }
    if not available:
        raise ValueError("No meshes for the requested anatomy")
    print("Extracting topology", flush=True)
    body.extract_topology(names=available, spacing_m=args.centerline_spacing_mm / 1000)
    for name in list(body.anatomy.structures):
        if name not in requested:
            del body.anatomy.structures[name]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".pipeline-") as temporary:
        folder = Path(temporary) / "bundle"
        body.export_patient_twin(folder)
        body.export_to_usd(folder / "human_body.usdc")
        shutil.move(str(folder), str(output))
    print(output / "patient_twin.yaml", flush=True)
    return output / "patient_twin.yaml"


if __name__ == "__main__":
    run(parser().parse_args())
