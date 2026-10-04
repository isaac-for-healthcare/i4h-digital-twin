# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example pipeline: import existing masks or meshes, attach CT, extract topology, and export.

``python examples/pipeline.py --source sample --anatomy aorta --output out/`` uses the bundled
s0011 masks and CT. ``--source segmentation --input labels.nii.gz --labels labels.json --ct
ct.nii.gz`` imports your own label volume; ``--source simple --input meshes.json`` imports
STL/OBJ meshes (``{"meshes": {name: file}, "mesh_to_body": {...}}``). ``import_body``
builds the ``HumanBody``; ``run`` extracts centerlines and writes a patient bundle plus a
standalone ``human_body.usdc``. For NV-Generate/NV-Segment inference use
``python -m patient_digital_twin``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from patient_digital_twin import HumanBody, SegmentationImporter
from patient_digital_twin.__main__ import class_names
from patient_digital_twin.importers import SimpleImporter
from patient_digital_twin.scan_volume import from_nifti

SAMPLE = Path(__file__).parent / "data/s0011"


def parser() -> argparse.ArgumentParser:
    """Build the example's CLI parser."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--source", choices=("sample", "segmentation", "simple"), default="sample"
    )
    cli.add_argument("--input", type=Path, help="Segmentation or mesh-map JSON")
    cli.add_argument("--labels", type=Path, help="Label map for segmentation input")
    cli.add_argument("--ct", type=Path, help="Matching CT for --source segmentation")
    cli.add_argument("--output", type=Path, required=True, help="New bundle directory")
    cli.add_argument("--anatomy", nargs="*", default=[], help="Optional anatomy subset")
    cli.add_argument("--centerline-spacing-mm", type=float, default=1.5)
    return cli


def import_body(args: argparse.Namespace) -> HumanBody:
    """Build a HumanBody and attach only the CT belonging to its anatomy."""
    ct = None
    if args.source == "sample":
        importer = SegmentationImporter(
            SAMPLE / "segmentations", names=class_names(args.anatomy or ["aorta"])
        )
        ct = SAMPLE / "ct.nii.gz"
    elif args.source == "segmentation":
        importer = SegmentationImporter(args.input, args.labels)
        ct = args.ct
    else:
        config = json.loads(args.input.read_text())
        meshes = {name: args.input.parent / value for name, value in config["meshes"].items()}
        importer = SimpleImporter(
            meshes,
            mesh_to_body=config.get("mesh_to_body"),
            body_to_imaging=config.get("body_to_imaging"),
        )
    body = HumanBody(importer.to_anatomy_collection())
    if ct is not None:
        body.attach_scan(from_nifti(ct), source_path=ct)
    return body


def run(args: argparse.Namespace) -> Path:
    """Validate options, import, extract vessel centerlines, and write the bundle; returns the manifest path."""
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    selected = set(class_names(args.anatomy)) if args.anatomy else set()
    if not np.isfinite(args.centerline_spacing_mm) or args.centerline_spacing_mm <= 0:
        raise ValueError("Centerline spacing must be positive and finite")
    if (args.source == "sample") == (args.input is not None):
        raise ValueError("--input is required for segmentation/simple and not accepted for sample")
    if (args.source == "segmentation") != (args.ct is not None):
        raise ValueError("--ct is required for, and only accepted with, segmentation input")
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
