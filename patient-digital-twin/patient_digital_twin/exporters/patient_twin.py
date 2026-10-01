# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export native scan-frame anatomy and a NumPy/YAML imaging artifact."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from ..artifacts import write_centerline
from ..geometry import rigid_transform, transform_points
from ..human import HumanBody
from ..imaging_to_mesh import mask_to_mesh
from ..structures import AnatomicalStructure, MeshGeometry
from ..topology import native_centerline
from .native import mask_on_scan, scan_for_body, scan_from_body_m
from .usd import _export_to_usd


def export_patient_twin(
    body,
    output,
    *,
    patient_id=None,
    vessel_names=(),
    world_from_patient_m=None,
    exterior="auto",
    skin_opacity=0.15,
):
    """Write schema 3: native arrays, full affine, source-frame geometry and units.

    No CT canonicalization, mask cleanup, resampling, or simulator placement is
    performed. An explicit world_from_patient_m is retained as an optional hint.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new patient-twin output directory: {output}")
    if exterior not in {"auto", "ct"}:
        raise ValueError("exterior must be auto or ct")
    scan = scan_for_body(body)
    if exterior == "ct" and scan is None:
        raise ValueError("CT exterior requires attached CT")
    for name in vessel_names:
        if (
            name not in body.anatomy.structures
            or body.anatomy.structures[name].is_empty
        ):
            raise ValueError(f"Missing or disabled vessel mesh: {name}")
    units = scan.meters_per_unit if scan is not None else 1.0
    frame = (
        scan.frame
        if scan is not None
        else "RAS"
        if body.anatomy.body_to_imaging is not None
        else "body"
    )
    placement = scan_from_body_m(body, scan)
    source_path = body.imaging.source_path if body.imaging is not None else None
    mask = (
        mask_on_scan(body, scan, vessel_names)
        if scan is not None and vessel_names
        else None
    )
    exterior_mesh = None
    if exterior == "ct":
        from scipy import ndimage

        envelope = ndimage.binary_fill_holes(scan.values_kji[::3, ::3, ::3] > -300)
        if envelope.any():
            points, faces = mask_to_mesh(envelope)
            points = transform_points(points * 3, scan.ijk_to_world) * units
            if np.linalg.det(scan.ijk_to_world[:3, :3]) < 0:
                faces = faces[:, ::-1]
            exterior_mesh = MeshGeometry(points, faces)
    snapshot = HumanBody(
        {
            name: AnatomicalStructure(
                name,
                structure.kind,
                structure.mesh.vertices,
                structure.mesh.faces,
                local_to_world=placement @ structure.local_to_body,
                enabled=structure.enabled,
                centerline=structure.centerline,
            )
            for name, structure in body.anatomy.structures.items()
            if structure.mesh.vertices is not None
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".patient-twin-") as temp:
        folder = Path(temp) / "bundle"
        artifacts = {"anatomy_usd": "patient_anatomy.usdc"}
        if scan is not None:
            scan.save(folder)
            artifacts.update(hu_volume="volume.npy", volume_metadata="volume.yaml")
        else:
            folder.mkdir()
        if mask is not None:
            np.save(folder / "vessel_mask.npy", mask.astype(np.uint8))
            points, edges, radii = native_centerline(mask, scan)
            artifacts.update(write_centerline(folder, points, edges, radii))
            artifacts["vessel_mask"] = "vessel_mask.npy"
        root_scale = np.eye(4)
        _export_to_usd(
            snapshot,
            folder / "patient_anatomy.usdc",
            root_transform=root_scale,
            meters_per_unit=units,
            exterior_mesh=exterior_mesh,
            skin_name="CT",
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
        centerlines = {}
        for index, (name, structure) in enumerate(body.anatomy.structures.items()):
            if structure.centerline is None:
                continue
            (folder / "centerlines").mkdir(exist_ok=True)
            relative = f"centerlines/{index}.npz"
            graph = structure.centerline
            points = (
                transform_points(graph.points, placement @ structure.local_to_body)
                / units
            )
            np.savez_compressed(
                folder / relative,
                points=points,
                edges=graph.edges,
                radii=graph.radii / units,
            )
            centerlines[name] = {
                "path": relative,
                "units": scan.metadata["output"]["world_unit"] if scan else "m",
                "coordinate_frame": frame,
                "local_to_patient": np.eye(4).tolist(),
            }
        manifest = {
            "schema_version": 3,
            "patient_id": patient_id
            or (Path(source_path).parent.name if source_path else "geometry"),
            "coordinate_frame": frame,
            "spatial_unit": scan.metadata["output"]["world_unit"] if scan else "m",
            "meters_per_unit": units,
            "anatomy": {
                "structures": structures,
                "exterior": {
                    "source": "CT",
                    "pose": "imaging",
                    "prim_path": "/HumanBody/Exterior/CT",
                }
                if exterior_mesh is not None
                else None,
                "missing_meshes": [
                    n
                    for n, s in body.anatomy.structures.items()
                    if s.mesh.vertices is None
                ],
            },
            "transforms": {"voxel_to_scan": scan.ijk_to_world.tolist()} if scan else {},
            "artifacts": artifacts,
            "centerlines": centerlines,
        }
        if world_from_patient_m is not None:
            manifest["transforms"]["world_from_patient_m"] = rigid_transform(
                world_from_patient_m
            ).tolist()
        (folder / "patient_twin.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )
        shutil.move(str(folder), str(output))
    return output / "patient_twin.yaml"
