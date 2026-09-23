# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the scan-frame artifact contract consumed by i4h-workflows PatientTwin."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from ..geometry import rigid_transform, transform_points
from ..human import HumanBody
from ..imaging_to_mesh import mask_to_mesh
from ..soma_body import PosedBody
from ..structures import AnatomicalStructure
from ..topology import extract_centerlines, voxelize_mesh
from .usd import _export_to_usd
from .utils import attached_ct, save_attenuation


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
    patient_id=None,
    vessel_names=(),
    world_from_patient_m=None,
    exterior="auto",
    skin_opacity=0.15,
    physics_root=None,
    soma_pose="scan",
):
    """Write a complete patient_twin.yaml bundle using original imaging placement.

    Source registration selects DICOM LPS; otherwise use the body frame.
    CT and SOMA are optional. An empty vessel_names
    tuple omits the composite navigation mask and centerline.
    CT and navigation artifacts retain original imaging placement. SOMA and
    rigid attached anatomy default to arms-down presentation; soma_pose="imaging"
    explicitly exports their original registration pose instead.
    CT supplies attenuation; the torso registration is unchanged by arm posing. Use exterior="ct" to request a CT envelope. Set exterior="soma"
    to require SOMA. All retained anatomy meshes are included, even hidden ones.
    Output must be a new directory to avoid stale bundles. Pass physics_root
    to also export OmniEndo/OmniSurg inputs from the retained patient anatomy.
    """
    from scipy import ndimage

    output = Path(output).expanduser().resolve()
    imaging = body.imaging
    source_path = imaging.source_path if imaging is not None else None
    body_to_imaging = (
        imaging.body_to_imaging if imaging is not None else body.anatomy.body_to_imaging
    )
    if output.exists():
        raise FileExistsError(f"Use a new patient-twin output directory: {output}")
    registered = body_to_imaging is not None
    if soma_pose not in {"scan", "imaging"}:
        raise ValueError("soma_pose must be scan or imaging")
    if exterior not in {"auto", "soma", "ct"}:
        raise ValueError("exterior must be auto, soma or ct")
    use_soma = exterior == "soma" or (
        exterior == "auto"
        and body.soma is not None
        and body.soma.soma_layer is not None
    )
    if use_soma and (body.soma is None or body.soma.soma_layer is None):
        raise ValueError("Attach SOMA before requesting its scan-frame exterior")
    vessel_names = tuple(vessel_names)
    for name in vessel_names:
        if (
            name not in body.anatomy.structures
            or body.anatomy.structures[name].is_empty
        ):
            raise ValueError(f"Missing or disabled vessel mesh: {name}")
    ct = None
    if imaging is not None:
        ct = attached_ct(imaging)
        if not np.allclose(
            np.asarray(ct.direction).reshape(3, 3), np.eye(3), atol=1e-4
        ):
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
    if exterior == "ct" and ct is None:
        raise ValueError("CT exterior requires attached CT")
    lps_from_body = (
        np.diag([-1.0, -1.0, 1.0, 1.0]) @ body_to_imaging if registered else np.eye(4)
    )
    if ct is not None and vessel_names:
        vessel_mask = np.zeros(ct.hu_zyx.shape, dtype=bool)
        for name in vessel_names:
            structure = body.anatomy.structures[name]
            points = transform_points(
                structure.mesh.vertices, lps_from_body @ structure.local_to_body
            )
            vessel_mask |= voxelize_mesh(points, structure.mesh.faces, **grid)
        vessel_mask = _largest(
            ndimage.binary_closing(
                vessel_mask, structure=np.ones((3, 3, 3)), iterations=2
            )
        )
        points, faces = mask_to_mesh(
            vessel_mask, spacing_zyx_mm=spacing, origin_xyz_mm=origin
        )
        graph = extract_centerlines(
            points.astype(float) * 0.001, faces, method="skeleton", **grid
        )
    if ct is not None:
        voxel_to_patient = np.diag([*spacing[::-1], 1.0])
        voxel_to_patient[:3, 3] = origin
    if world_from_patient_m is None and not registered:
        world = np.eye(4)
    elif world_from_patient_m is None:
        world = np.eye(4)
        world[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        center = (
            origin + (np.asarray(ct.hu_zyx.shape[::-1]) - 1) * spacing[::-1] / 2
            if ct is not None
            else np.zeros(3)
        )
        world[:3, 3] = [0, 0, 0.85] - world[:3, :3] @ (center * 0.001)
    else:
        world = rigid_transform(world_from_patient_m)
    if use_soma:
        import torch

        # Bindings retain imaging placement, while presentation defaults to arms
        # down. Evaluate without the synchronization hook or mutating the body.
        parameters = body.soma.soma_parameters
        if soma_pose == "scan":
            reference = parameters["poses"]
            parameters["poses"] = torch.as_tensor(
                body.soma.scan_pose, dtype=reference.dtype, device=reference.device
            )
        with torch.no_grad():
            scan_skin = PosedBody.from_output(
                body.soma.soma_layer, body.soma.soma_layer.forward(**parameters)
            )
        skin_points = transform_points(
            scan_skin.vertices, lps_from_body @ np.linalg.inv(body.soma.body_to_soma)
        )
        skin_faces, skin_name = scan_skin.faces, "SOMA"
    elif ct is not None and exterior == "ct":
        envelope = _largest(
            ndimage.binary_closing(ct.hu_zyx[::3, ::3, ::3] > -300, iterations=2)
        )
        envelope = ndimage.binary_fill_holes(envelope)
        skin_points, skin_faces = mask_to_mesh(
            envelope, spacing_zyx_mm=spacing * 3, origin_xyz_mm=origin
        )
        skin_points = skin_points.astype(float) * 0.001
        skin_name = "CT"
    else:
        skin_name = None

    def structure_transform(structure):
        if use_soma and soma_pose == "scan" and structure.local_to_anchor is not None:
            return (
                lps_from_body
                @ np.linalg.inv(body.soma.body_to_soma)
                @ scan_skin.transforms[structure.anchor_joint]
                @ structure.local_to_anchor
            )
        return lps_from_body @ structure.local_to_body

    snapshot = HumanBody(
        {
            name: AnatomicalStructure(
                name,
                structure.kind,
                structure.mesh.vertices,
                structure.mesh.faces,
                local_to_world=structure_transform(structure),
                enabled=structure.enabled,
                centerline=structure.centerline,
            )
            for name, structure in body.anatomy.structures.items()
            if structure.mesh.vertices is not None
        }
    )
    from ..soma_body import SomaRepresentation

    if skin_name is not None:
        snapshot.soma = SomaRepresentation(snapshot.anatomy)
        snapshot.soma.soma_body = PosedBody(skin_points, skin_faces, {}, {})
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".patient-twin-") as temp:
        folder = Path(temp) / "bundle"
        folder.mkdir()
        if ct is not None:
            save_attenuation(ct, folder, source=source_path or "numpy")
            np.save(folder / "hu_volume.npy", ct.hu_zyx.astype(np.float32))
        if ct is not None and vessel_names:
            np.save(folder / "vessel_mask.npy", vessel_mask.astype(np.uint8))
            np.save(folder / "centerline_points_mm.npy", graph.points * 1000)
            np.save(folder / "centerline_radii_mm.npy", graph.radii * 1000)
            np.save(folder / "centerline_edges.npy", graph.edges)
        _export_to_usd(
            snapshot,
            folder / "patient_anatomy.usdc",
            root_transform=np.eye(4),
            skin_name=skin_name or "SOMA",
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
            "patient_id": patient_id
            or (Path(source_path).parent.name if source_path else "geometry"),
            "coordinate_frame": "DICOM_LPS" if registered else "body",
            "anatomy": {
                "exterior": {
                    "source": skin_name,
                    "pose": soma_pose if use_soma else "imaging",
                    "prim_path": f"/HumanBody/Exterior/{skin_name}",
                }
                if skin_name is not None
                else None,
                "structures": structures,
                "missing_meshes": [
                    name
                    for name, s in body.anatomy.structures.items()
                    if s.mesh.vertices is None
                ],
            },
            "transforms": {
                **(
                    {"voxel_to_patient_mm": voxel_to_patient.tolist()}
                    if ct is not None
                    else {}
                ),
                "world_from_patient_m": world.tolist(),
            },
            "artifacts": {
                **(
                    {
                        "attenuation_volume": "mu_volume.npy",
                        "volume_metadata": "metadata.json",
                        "hu_volume": "hu_volume.npy",
                    }
                    if ct is not None
                    else {}
                ),
                **(
                    {
                        "vessel_mask": "vessel_mask.npy",
                        "centerline_points": "centerline_points_mm.npy",
                        "centerline_edges": "centerline_edges.npy",
                        "centerline_radii": "centerline_radii_mm.npy",
                    }
                    if ct is not None and vessel_names
                    else {}
                ),
                "anatomy_usd": "patient_anatomy.usdc",
            },
        }
        centerlines = {}
        for index, (name, structure) in enumerate(body.anatomy.structures.items()):
            if structure.centerline is None:
                continue
            (folder / "centerlines").mkdir(exist_ok=True)
            relative = f"centerlines/{index}.npz"
            graph = structure.centerline
            np.savez_compressed(
                folder / relative,
                points=graph.points,
                edges=graph.edges,
                radii=graph.radii,
            )
            centerlines[name] = {
                "path": relative,
                "units": "m",
                "coordinate_frame": "structure_local",
                "local_to_patient": structure_transform(structure).tolist(),
            }
        manifest["centerlines"] = centerlines
        (folder / "patient_twin.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )
        if physics_root is not None:
            from .physics_export import export_physics_examples

            physics_manifest = export_physics_examples(
                folder / "patient_twin.yaml",
                folder / "simulation",
                physics_root=physics_root,
            )
            manifest["physics_examples"] = yaml.safe_load(physics_manifest.read_text())[
                "physics_examples"
            ]
            manifest["physics_examples"]["source_patient_twin"] = "patient_twin.yaml"
            for demo in manifest["physics_examples"]["demos"].values():
                demo["config"] = "simulation/" + demo["config"]
            (folder / "patient_twin.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=False)
            )
        shutil.move(str(folder), str(output))
    return output / "patient_twin.yaml"
