# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read/write anatomy and imaging caches, independent of the export pipeline."""

import json
from pathlib import Path

import numpy as np
from patient_digital_twin import AnatomicalStructure, HumanBody, Kind


def save_body(body, folder):
    """Store numeric geometry and JSON registration without pickle."""
    arrays, structures = {}, {}
    for name, s in body.anatomy.structures.items():
        if s.mesh.vertices is None:
            continue
        arrays[name + "_vertices"] = s.mesh.vertices
        arrays[name + "_faces"] = s.mesh.faces
        structures[name] = {
            "kind": s.kind.value,
            "enabled": s.enabled,
            "local_to_body": s.local_to_body.tolist(),
        }
    imaging_metadata = None
    if body.imaging is not None:
        arrays["imaging_volume"] = body.imaging.volume
        imaging_metadata = {
            "voxel_to_imaging": body.imaging.voxel_to_imaging.tolist(),
            "body_to_imaging": body.imaging.body_to_imaging.tolist(),
            "source_path": body.imaging.source_path,
            "modality": body.imaging.modality,
        }
    np.savez_compressed(folder / "body.npz", **arrays)
    metadata = {
        "structures": structures,
        "imaging": imaging_metadata,
        "body_to_imaging": body.anatomy.body_to_imaging.tolist()
        if body.anatomy.body_to_imaging is not None
        else None,
    }
    (folder / "body.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_body(folder):
    """Reload cached anatomy and attached imaging."""
    folder = Path(folder)
    metadata = json.loads((folder / "body.json").read_text())
    with np.load(folder / "body.npz", allow_pickle=False) as arrays:
        structures = {
            name: AnatomicalStructure(
                name,
                Kind(item["kind"]),
                arrays[name + "_vertices"],
                arrays[name + "_faces"],
                local_to_body=np.asarray(item["local_to_body"], dtype=float),
                enabled=item["enabled"],
            )
            for name, item in metadata["structures"].items()
        }
        volume = (
            arrays["imaging_volume"] if metadata.get("imaging") is not None else None
        )
    body = HumanBody(structures)
    body.anatomy.body_to_imaging = metadata["body_to_imaging"]
    if volume is not None:
        body.AttachImaging(volume, **metadata["imaging"])
    return body
