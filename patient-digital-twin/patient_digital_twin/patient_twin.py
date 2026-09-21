# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the scan-frame artifact contract consumed by i4h-workflows PatientTwin."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from .geometry import rigid_transform, transform_points
from .human import HumanBody
from .imaging_to_mesh import mask_to_mesh
from .soma_body import PosedBody
from .structures import AnatomicalStructure
from .topology import extract_centerlines, voxelize_mesh
from .usd import _export_to_usd


def _largest(mask):
    from scipy import ndimage

    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if not count:
        raise ValueError("No foreground in exported volume")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == sizes.argmax()


def export_patient_twin(
    body,
    output,
    *,
    ct_path,
    patient_id=None,
    vessel_names=("aorta", "iliac_artery_left", "iliac_artery_right"),
    world_from_patient_m=None,
    exterior="auto",
    skin_opacity=0.15,
):
    """Write a complete patient_twin.yaml bundle using original imaging placement.

    Requires body_to_imaging (NIfTI RAS meters), CT, and enabled vessel meshes.
    Posed anatomy is deliberately not used: the source CT cannot follow posing.
    CT supplies attenuation; attached SOMA supplies the exterior in its original
    registration pose. Without SOMA, auto uses a CT envelope. Set exterior="soma"
    to require SOMA. All retained anatomy meshes are included, even hidden ones.
    Output must be a new directory to avoid stale bundles.
    """
    try:
        from vasculature_digital_twin import (
            HuToMuMapping,
            PreprocessingSettings,
            VolumePreprocessor,
        )
        from vasculature_digital_twin.ct.dicom_ingest import load_nifti_hu
    except ImportError as exc:
        raise ImportError(
            "Patient-twin export requires the optional vasculature-digital-twin package with its io extra"
        ) from exc
    from scipy import ndimage

    output = Path(output).expanduser().resolve()
    ct_path = Path(ct_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new patient-twin output directory: {output}")
    if body.body_to_imaging is None:
        raise ValueError(
            "Patient-twin export requires body_to_imaging in NIfTI RAS meters"
        )
    if exterior not in {"auto", "soma", "ct"}:
        raise ValueError("exterior must be auto, soma or ct")
    use_soma = exterior == "soma" or (
        exterior == "auto" and body.soma_layer is not None
    )
    if use_soma and body.soma_layer is None:
        raise ValueError("Attach SOMA before requesting its scan-frame exterior")
    vessel_names = tuple(vessel_names)
    if not vessel_names:
        raise ValueError("Select at least one vessel structure")
    for name in vessel_names:
        if name not in body.structures or body.structures[name].is_empty:
            raise ValueError(f"Missing or disabled vessel mesh: {name}")
    ct = load_nifti_hu(ct_path)
    if not np.allclose(np.asarray(ct.direction).reshape(3, 3), np.eye(3), atol=1e-4):
        raise ValueError(
            "Resample oblique CT to patient axes before patient-twin export"
        )
    if not np.isfinite(ct.hu_zyx).all():
        raise ValueError("CT contains non-finite intensities")
    spacing, origin = np.asarray(ct.spacing_zyx_mm), np.asarray(ct.origin_xyz_mm)
    grid = {
        "shape_zyx": ct.hu_zyx.shape,
        "spacing_zyx_m": spacing * 0.001,
        "origin_xyz_m": origin * 0.001,
    }
    lps_from_body = np.diag([-1.0, -1.0, 1.0, 1.0]) @ body.body_to_imaging
    vessel_mask = np.zeros(ct.hu_zyx.shape, dtype=bool)
    for name in vessel_names:
        structure = body.structures[name]
        points = transform_points(
            structure.mesh.vertices, lps_from_body @ structure.local_to_body
        )
        vessel_mask |= voxelize_mesh(points, structure.mesh.faces, **grid)
    vessel_mask = _largest(
        ndimage.binary_closing(vessel_mask, structure=np.ones((3, 3, 3)), iterations=2)
    )
    points, faces = mask_to_mesh(
        vessel_mask, spacing_zyx_mm=spacing, origin_xyz_mm=origin
    )
    graph = extract_centerlines(
        points.astype(float) * 0.001, faces, method="skeleton", **grid
    )
    voxel_to_patient = np.diag([*spacing[::-1], 1.0])
    voxel_to_patient[:3, 3] = origin
    if world_from_patient_m is None:
        world = np.eye(4)
        world[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        center = origin + (np.asarray(ct.hu_zyx.shape[::-1]) - 1) * spacing[::-1] / 2
        world[:3, 3] = [0, 0, 0.85] - world[:3, :3] @ (center * 0.001)
    else:
        world = rigid_transform(world_from_patient_m)
    if use_soma:
        import torch

        # Evaluate the saved attachment pose, not the current display pose.
        # Direct forward bypasses the synchronization hook and leaves body intact.
        with torch.no_grad():
            scan_skin = PosedBody.from_output(
                body.soma_layer, body.soma_layer.forward(**body.soma_parameters)
            )
        skin_points = transform_points(
            scan_skin.vertices, lps_from_body @ np.linalg.inv(body.body_to_soma)
        )
        skin_faces, skin_name = scan_skin.faces, "SOMA"
    else:
        envelope = _largest(
            ndimage.binary_closing(ct.hu_zyx[::3, ::3, ::3] > -300, iterations=2)
        )
        envelope = ndimage.binary_fill_holes(envelope)
        skin_points, skin_faces = mask_to_mesh(
            envelope, spacing_zyx_mm=spacing * 3, origin_xyz_mm=origin
        )
        skin_points = skin_points.astype(float) * 0.001
        skin_name = "CT"
    snapshot = HumanBody(
        {
            name: AnatomicalStructure(
                name,
                structure.kind,
                structure.mesh.vertices,
                structure.mesh.faces,
                local_to_world=lps_from_body @ structure.local_to_body,
                enabled=structure.enabled,
            )
            for name, structure in body.structures.items()
            if structure.mesh.vertices is not None
        }
    )
    snapshot.soma.soma_body = PosedBody(skin_points, skin_faces, {}, {})
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".patient-twin-") as temp:
        folder = Path(temp) / "bundle"
        folder.mkdir()
        mapping = HuToMuMapping(
            control_points=(
                (-1000.0, 0.0),
                (-300.0, 0.0),
                (100.0, 0.0008),
                (300.0, 0.0028),
                (500.0, 0.006),
                (900.0, 0.009),
                (1500.0, 0.012),
                (3000.0, 0.02),
                (8000.0, 0.044),
            )
        )
        processor = VolumePreprocessor(
            hu_volume=ct.hu_zyx,
            spacing_zyx_mm=ct.spacing_zyx_mm,
            origin_xyz_mm=ct.origin_xyz_mm,
            source=str(ct_path),
            settings=PreprocessingSettings(hu_to_mu=mapping, clip_hu=False),
            anatomical_frame=ct.anatomical_frame,
            source_orientation=ct.source_orientation,
            direction=ct.direction,
        )
        volume = processor.preprocess()
        volume.metadata.hu_to_mu = {"preset": "interventional", **mapping.to_dict()}
        volume.save(folder)
        np.save(folder / "hu_volume.npy", ct.hu_zyx.astype(np.float32))
        np.save(folder / "vessel_mask.npy", vessel_mask.astype(np.uint8))
        np.save(folder / "centerline_points_mm.npy", graph.points * 1000)
        np.save(folder / "centerline_radii_mm.npy", graph.radii * 1000)
        np.save(folder / "centerline_edges.npy", graph.edges)
        _export_to_usd(
            snapshot,
            folder / "patient_anatomy.usdc",
            root_transform=np.eye(4),
            skin_name=skin_name,
            skin_opacity=skin_opacity,
        )
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(folder / "patient_anatomy.usdc"))
        structures = {
            prim.GetCustomDataByKey("anatomy:name"): {
                "prim_path": str(prim.GetPath()),
                "kind": prim.GetCustomDataByKey("anatomy:kind"),
                "enabled": UsdGeom.Imageable(prim).ComputeVisibility() != "invisible",
            }
            for prim in stage.Traverse()
            if prim.GetCustomDataByKey("anatomy:name") is not None
        }
        manifest = {
            "schema_version": 1,
            "patient_id": patient_id or ct_path.parent.name,
            "coordinate_frame": "DICOM_LPS",
            "anatomy": {
                "exterior": {
                    "source": skin_name,
                    "prim_path": f"/HumanBody/Exterior/{skin_name}",
                },
                "structures": structures,
                "missing_meshes": [
                    name
                    for name, s in body.structures.items()
                    if s.mesh.vertices is None
                ],
            },
            "transforms": {
                "voxel_to_patient_mm": voxel_to_patient.tolist(),
                "world_from_patient_m": world.tolist(),
            },
            "artifacts": {
                "attenuation_volume": "mu_volume.npy",
                "volume_metadata": "metadata.json",
                "hu_volume": "hu_volume.npy",
                "vessel_mask": "vessel_mask.npy",
                "centerline_points": "centerline_points_mm.npy",
                "centerline_edges": "centerline_edges.npy",
                "centerline_radii": "centerline_radii_mm.npy",
                "anatomy_usd": "patient_anatomy.usdc",
            },
        }
        (folder / "patient_twin.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )
        shutil.move(str(folder), str(output))
    return output / "patient_twin.yaml"
