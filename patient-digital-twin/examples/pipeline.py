# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import anatomy, attach matching CT, extract topology, optionally attach SOMA, export."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from patient_digital_twin import HumanBody, SegmentationImporter
from patient_digital_twin.catalog import CATALOG
from patient_digital_twin.importers import (
    NVGenerateImporter,
    NVSegmentImporter,
    SimpleImporter,
    TotalSegmentatorImporter,
)

SAMPLE = Path(__file__).parent / "data/nv_ct_high_resolution"


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--source",
        choices=(
            "sample",
            "segmentation",
            "nvgenerate",
            "nvsegment",
            "totalsegmentator",
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
    cli.add_argument(
        "--soma",
        action="store_true",
        help="Attach SOMA after topology extraction (not for simple input)",
    )
    cli.add_argument(
        "--parameters",
        type=Path,
        help="Optional SOMA registration JSON; requires --soma",
    )
    cli.add_argument("--source-root", type=Path, help="NVGenerate checkout")
    cli.add_argument("--bundle-root", type=Path, help="NVSegment bundle")
    cli.add_argument(
        "--python", dest="python_executable", help="NVIDIA backend interpreter"
    )
    cli.add_argument("--device", default="gpu", help="TotalSegmentator device")
    cli.add_argument("--centerline-spacing-mm", type=float, default=1.5)
    return cli


def import_body(args):
    """Build a HumanBody and attach only the CT belonging to its anatomy."""
    source = args.source
    path = args.input.expanduser().resolve() if args.input else None
    ct = None
    if source == "sample":
        importer = SegmentationImporter(
            SAMPLE / "segmentation.nii.gz", SAMPLE / "labels.json"
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
        importer = TotalSegmentatorImporter(path, modality="CT", device=args.device)
        ct = path
    body = HumanBody(importer.to_anatomy_collection())
    if source == "sample":
        body.anatomy.configure(SAMPLE / "anatomy.yaml")
    if source == "nvgenerate":
        body.AttachImaging(
            importer.ct_volume_zyx, voxel_to_imaging=importer.ct_voxel_to_imaging
        )
    elif ct is not None:
        image = nib.load(str(ct))
        body.AttachImaging(
            image.get_fdata(dtype=np.float32).transpose(2, 1, 0),
            voxel_to_imaging=SegmentationImporter._affine_m(image),
            source_path=ct,
        )
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
    if args.source == "simple" and (args.soma or args.parameters):
        raise ValueError("Simple input skips CT and SOMA attachment")
    if args.parameters and not args.soma:
        raise ValueError("--parameters requires --soma")

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
    if args.soma:
        parameters = args.parameters or (
            SAMPLE / "viewer_parameters.json" if args.source == "sample" else None
        )
        options = json.loads(parameters.expanduser().read_text()) if parameters else {}
        body.AttachExternalBody(**options)
    # Keep all bones available for matching before applying the export subset.
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
