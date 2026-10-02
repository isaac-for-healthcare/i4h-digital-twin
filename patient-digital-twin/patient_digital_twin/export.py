# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exporters: write a ``HumanBody`` as OpenUSD, as an i4h-workflows bundle, or as raw arrays.

Usually reached through ``HumanBody.export_to_usd`` and ``HumanBody.export_patient_twin``.

- ``export_to_usd``: one standalone stage with anatomy in its current pose, centerlines
  as custom prim attributes, and attached CT as native HU attributes. The stage uses the
  scan's units (meters without CT), Z-up, and default prim ``/HumanBody``.
- ``export_patient_twin``: a new schema-3 bundle directory: ``patient_twin.yaml``
  (manifest), ``patient_anatomy.usdc`` (anatomy in the scan's physical frame), and with
  CT ``volume.npy``/``volume.yaml``; ``vessel_names`` adds ``vessel_mask.npy`` and
  ``centerline_{points,edges,radii}.npy`` on the CT grid for navigation.
- ``write_artifacts``: just the native CT and optional vessel mask/centerline arrays.
- ``write_usd``: the shared stage writer behind both exporters.

Nothing here resamples, reorients, or recenters the source scan. Directory outputs are
written to a sibling staging folder and moved into place only when complete.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from .geometry import (
    mask_to_mesh,
    native_centerline,
    rigid_transform,
    transform_points,
    validate_triangles,
    voxelize_mesh,
)

if TYPE_CHECKING:
    from .body import HumanBody
    from .scan_volume import ScanVolume

USD_SUFFIXES = {".usd", ".usda", ".usdc"}


def _scan(body: HumanBody) -> ScanVolume | None:
    """The attached ScanVolume, if any."""
    return None if body.imaging is None else body.imaging.scan


def _scan_from_body(body: HumanBody, scan: ScanVolume | None) -> NDArray[np.float64]:
    """Body frame (meters) to the scan's physical frame (still meters)."""
    registration = body.anatomy.body_to_imaging if body.imaging is None else body.imaging.body_to_imaging
    if registration is None:
        return np.eye(4)
    flip = np.diag([-1.0, -1.0, 1.0, 1.0]) if scan is not None and scan.frame == "LPS" else np.eye(4)
    return flip @ registration


def _write_scan(scan: ScanVolume, folder: Path, vessel_mask: ArrayLike | None = None) -> dict[str, str]:
    """Native HU volume + YAML, plus an optional scan-grid vessel mask and its centerline arrays."""
    scan.save(folder)
    paths = {"hu_volume": "volume.npy", "volume_metadata": "volume.yaml"}
    if vessel_mask is not None:
        mask = np.asarray(vessel_mask)
        if mask.shape != scan.values.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("Vessel mask must be binary and match the scan grid")
        names = ("centerline_points", "centerline_edges", "centerline_radii")
        arrays: dict[str, np.ndarray] = {**dict(zip(names, native_centerline(mask, scan))), "vessel_mask": mask.astype(np.uint8)}
        for name, values in arrays.items():
            np.save(Path(folder) / f"{name}.npy", values, allow_pickle=False)
            paths[name] = f"{name}.npy"
    return paths


def _publish(output: str | Path, prefix: str, write: Callable[[Path], Any]) -> Path:
    """Run write(folder) in a sibling staging folder, then move it to a new output path."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=prefix) as temp:
        write(Path(temp) / "out")
        shutil.move(str(Path(temp) / "out"), str(output))
    return output


def write_artifacts(scan: ScanVolume, output: str | Path, *, vessel_mask: ArrayLike | None = None) -> Path:
    """Write native HU ``volume.npy`` + ``volume.yaml`` and an optional vessel mask with its centerline.

    Args:
        scan: Source CT, e.g. ``scan_volume.from_nifti("ct.nii.gz")``.
        output: New directory (must not exist); written atomically.
        vessel_mask: Binary array with ``scan.values``' shape and axis order.

    Returns:
        The output directory.
    """
    return _publish(output, ".ct-artifacts-", lambda folder: _write_scan(scan, folder, vessel_mask))


def _vessel_mask(body: HumanBody, scan: ScanVolume, names: Sequence[str]) -> NDArray[np.bool_]:
    """Requested vessels on the scan grid: source labels, or rasterized meshes without them."""
    anatomy, imaging = body.anatomy, body.imaging
    labels, labels_to_ras = anatomy.source_segmentation, anatomy.source_voxel_to_ras_m
    if imaging is None:
        raise ValueError("Vessel masks require attached CT")
    if labels is not None:
        if labels_to_ras is None or labels.shape != scan.values_kji.shape or not np.allclose(
            labels_to_ras, scan.ijk_to_ras_m, atol=1e-9, rtol=1e-6
        ):
            raise ValueError("Segmentation and attached scan must share the same physical grid")
        mask = np.isin(labels, [i for i, n in anatomy.source_label_names.items() if n in names])
    else:
        mask = np.zeros(scan.values_kji.shape, bool)
        for name in names:
            structure = anatomy.structures[name]
            vertices, faces = structure.mesh.vertices, structure.mesh.faces
            if vertices is None or faces is None:
                raise ValueError(f"Missing vessel mesh: {name}")
            to_ijk = np.linalg.inv(scan.ijk_to_ras_m) @ imaging.body_to_imaging @ structure.local_to_body
            mask |= voxelize_mesh(
                transform_points(vertices, to_ijk), faces,
                shape_zyx=mask.shape, spacing_zyx_m=(1.0, 1.0, 1.0), origin_xyz_m=(0.0, 0.0, 0.0),
            )
    if not mask.any():
        raise ValueError("Selected vessels have no foreground in the scan")
    return mask.transpose(["kji".index(c) for c in scan.array_axes])


def export_to_usd(body: HumanBody, path: str | Path) -> Path:
    """Write a standalone USD of ``body``: current pose, stored centerlines, and attached CT (native HU).

    The stage uses the scan's units (meters without CT), Z-up, and default prim ``/HumanBody``.
    An existing file is replaced only after the new layer is complete. Returns the resolved path.

    Example:
        ``export_to_usd(body, "patient.usdc")`` (same as ``body.export_to_usd("patient.usdc")``)
    """
    scan = _scan(body)
    units = scan.meters_per_unit if scan is not None else 1.0
    root = _scan_from_body(body, scan).copy()
    root[:3, 3] /= units
    path = Path(path).expanduser().resolve()
    write_usd(body, path, units=units, root=root, scan=scan)
    return path


def write_usd(
    body: HumanBody,
    path: str | Path,
    *,
    units: float = 1.0,
    root: ArrayLike | None = None,
    placement: NDArray[np.floating] | None = None,
    scan: ScanVolume | None = None,
    exterior: tuple[ArrayLike, ArrayLike] | None = None,
    skin_opacity: float = 0.15,
) -> dict[str, str]:
    """Write a USD stage and return ``{structure name: prim path}``.

    Args:
        units: Stage meters-per-unit; geometry (in meters) is divided by it.
        root: 4x4 transform on ``/HumanBody`` (in stage units).
        placement: If given, each structure is placed at ``placement @ local_to_body``
            instead of its ``local_to_world``.
        scan: CT to embed under ``/HumanBody/Imaging/CT``.
        exterior: Optional ``(vertices, faces)`` CT envelope in stage-frame meters.
    """
    try:
        from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, Vt
    except ImportError as exc:
        raise ImportError("USD export is optional; install with: pip install usd-core") from exc
    path = Path(path)
    if path.suffix.lower() not in USD_SUFFIXES:
        raise ValueError("Use a .usd, .usda, or .usdc output file")
    if not np.isfinite(skin_opacity) or not 0 <= skin_opacity <= 1:
        raise ValueError("skin_opacity must be between 0 and 1")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, units)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    xform = UsdGeom.Xform.Define(stage, "/HumanBody")
    stage.SetDefaultPrim(xform.GetPrim())
    xform.AddTransformOp().Set(Gf.Matrix4d((np.eye(4) if root is None else np.asarray(root)).T.tolist()))
    for scope in ("Anatomy", "Looks") + (("Exterior",) if exterior is not None else ()):
        UsdGeom.Scope.Define(stage, f"/HumanBody/{scope}")

    def custom(prim: Any, name: str, kind: str, value: Any) -> None:
        """Author a custom attribute of Sdf type ``kind`` (e.g. ``"FloatArray"``) on ``prim``."""
        prim.CreateAttribute(name, getattr(Sdf.ValueTypeNames, kind), custom=True).Set(value)

    def mesh(prim_path: str, vertices: ArrayLike, faces: ArrayLike, color: Sequence[float], opacity: float,
             matrix: ArrayLike | None = None, enabled: bool = True) -> Any:
        """Define one shaded mesh prim (vertices in meters) and return it."""
        meters, triangles = validate_triangles(vertices, faces, name=prim_path)
        points = (meters / units).astype(np.float32)
        result = UsdGeom.Mesh.Define(stage, prim_path)
        result.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        result.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
        result.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.astype(np.int32).ravel()))
        result.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        result.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        result.CreateDoubleSidedAttr(True)
        result.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)])))
        result.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        result.CreateDisplayOpacityAttr([opacity])
        result.CreateVisibilityAttr(UsdGeom.Tokens.inherited if enabled else UsdGeom.Tokens.invisible)
        if matrix is not None:
            pose = rigid_transform(matrix)
            pose[:3, 3] /= units
            result.AddTransformOp().Set(Gf.Matrix4d(pose.T.tolist()))
        material = UsdShade.Material.Define(stage, "/HumanBody/Looks/" + result.GetPrim().GetName())
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Surface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(result.GetPrim()).Bind(material)
        return result.GetPrim()

    if exterior is not None:
        mesh("/HumanBody/Exterior/CT", *exterior, (0.72, 0.77, 0.84), skin_opacity).SetDisplayName("CT exterior")
    prims: dict[str, str] = {}
    for name, structure in body.anatomy.structures.items():
        vertices, faces = structure.mesh.vertices, structure.mesh.faces
        if vertices is None or faces is None:
            continue
        identifier = candidate = Tf.MakeValidIdentifier(name)
        number = 2
        while "/HumanBody/Anatomy/" + candidate in prims.values():
            candidate, number = f"{identifier}_{number}", number + 1
        prims[name] = "/HumanBody/Anatomy/" + candidate
        digest = hashlib.sha256(name.encode()).digest()
        color = (0.88, 0.84, 0.69) if structure.kind.value == "bone" else tuple(0.25 + 0.65 * v / 255 for v in digest[:3])
        matrix = structure.local_to_world if placement is None else placement @ structure.local_to_body
        prim = mesh(prims[name], vertices, faces, color, 1.0, matrix, structure.enabled)
        prim.SetDisplayName(name)
        prim.SetCustomDataByKey("anatomy:name", name)
        prim.SetCustomDataByKey("anatomy:kind", structure.kind.value)
        if (graph := structure.centerline) is not None:
            custom(prim, "centerline:points", "Point3fArray", Vt.Vec3fArray.FromNumpy(np.asarray(graph.points / units, np.float32)))
            custom(prim, "centerline:edges", "Int2Array", Vt.Vec2iArray.FromNumpy(np.asarray(graph.edges, np.int32)))
            custom(prim, "centerline:radii", "FloatArray", Vt.FloatArray.FromNumpy(np.asarray(graph.radii / units, np.float32)))
            custom(prim, "centerline:coordinateFrame", "Token", "structure_local")
    if scan is not None:
        UsdGeom.Scope.Define(stage, "/HumanBody/Imaging")
        ct = UsdGeom.Scope.Define(stage, "/HumanBody/Imaging/CT").GetPrim()
        output = scan.metadata["output"]
        custom(ct, "ct:hu", "FloatArray", Vt.FloatArray.FromNumpy(scan.values.ravel(order="C")))
        custom(ct, "ct:shape", "Int3", Gf.Vec3i(*scan.values.shape))
        custom(ct, "ct:arrayOrder", "Token", scan.array_axes)
        custom(ct, "ct:arrayIndexToScan", "Matrix4d", Gf.Matrix4d(np.asarray(output["array_index_to_world"]).T.tolist()))
        custom(ct, "ct:coordinateFrame", "Token", scan.frame)
        custom(ct, "ct:spatialUnit", "Token", output["world_unit"])
        custom(ct, "ct:units", "Token", "HU")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Finish the layer before replacing a previous export.
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        if not stage.GetRootLayer().Export(str(temporary)):
            raise RuntimeError(f"USD export failed: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return prims


def export_patient_twin(
    body: HumanBody,
    output: str | Path,
    *,
    patient_id: str | None = None,
    vessel_names: Sequence[str] = (),
    world_from_patient_m: ArrayLike | None = None,
    ct_exterior: bool = False,
    skin_opacity: float = 0.15,
) -> Path:
    """Write a schema-3 patient bundle directory and return its ``patient_twin.yaml`` path.

    Geometry uses the attached scan's physical frame and units (or the body/RAS frame in
    meters without CT). No CT canonicalization, mask cleanup, resampling, or simulator
    placement is performed. Structure centerlines stay on their USD prims.

    Args:
        output: New directory (must not exist).
        patient_id: Manifest ID; defaults to the CT source folder name or ``"geometry"``.
        vessel_names: Vessels to rasterize on the CT grid, with navigation centerline arrays.
        world_from_patient_m: Optional rigid simulator placement hint stored in the manifest.
        ct_exterior: Add a CT-derived skin envelope at ``/HumanBody/Exterior/CT``.
        skin_opacity: Envelope opacity in [0, 1].

    Example:
        ``export_patient_twin(body, "bundle", vessel_names=["aorta"], ct_exterior=True)``
    """
    if Path(output).expanduser().exists():
        raise FileExistsError(f"Use a new patient-twin output directory: {output}")
    scan = _scan(body)
    if ct_exterior and scan is None:
        raise ValueError("CT exterior requires attached CT")
    for name in vessel_names:
        if name not in body.anatomy.structures or body.anatomy.structures[name].is_empty:
            raise ValueError(f"Missing or disabled vessel mesh: {name}")
    units = scan.meters_per_unit if scan is not None else 1.0
    frame = scan.frame if scan is not None else "RAS" if body.anatomy.body_to_imaging is not None else "body"
    mask = _vessel_mask(body, scan, vessel_names) if scan is not None and vessel_names else None
    exterior = None
    if ct_exterior and scan is not None:
        from scipy import ndimage

        # Keep only the body: the table and air cavities would otherwise add nested shells.
        labels, count = ndimage.label(ndimage.binary_closing(scan.values_kji[::3, ::3, ::3] > -300, iterations=2))
        envelope = ndimage.binary_fill_holes(labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1) if count else None
        if envelope is not None:
            points, faces = mask_to_mesh(envelope)
            points = transform_points(points * 3, scan.ijk_to_world) * units
            exterior = (points, faces[:, ::-1] if np.linalg.det(scan.ijk_to_world[:3, :3]) < 0 else faces)
    source_path = None if body.imaging is None else body.imaging.source_path

    def write(folder: Path) -> None:
        """Write every bundle file into the staging ``folder``."""
        artifacts = {"anatomy_usd": "patient_anatomy.usdc"}
        if scan is not None:
            artifacts.update(_write_scan(scan, folder, mask))
        else:
            folder.mkdir()
        prims = write_usd(body, folder / "patient_anatomy.usdc", units=units, scan=None,
                          placement=_scan_from_body(body, scan), exterior=exterior, skin_opacity=skin_opacity)
        structures = body.anatomy.structures
        manifest: dict[str, Any] = {
            "schema_version": 3,
            "patient_id": patient_id or (Path(source_path).parent.name if source_path else "geometry"),
            "coordinate_frame": frame,
            "spatial_unit": scan.metadata["output"]["world_unit"] if scan is not None else "m",
            "meters_per_unit": units,
            "anatomy": {
                "structures": {
                    n: {"prim_path": p, "kind": structures[n].kind.value, "enabled": structures[n].enabled}
                    for n, p in prims.items()
                },
                "exterior": None if exterior is None
                else {"source": "CT", "pose": "imaging", "prim_path": "/HumanBody/Exterior/CT"},
                "missing_meshes": [n for n, s in structures.items() if s.mesh.vertices is None],
            },
            "transforms": {} if scan is None else {"voxel_to_scan": scan.ijk_to_world.tolist()},
            "artifacts": artifacts,
        }
        if world_from_patient_m is not None:
            manifest["transforms"]["world_from_patient_m"] = rigid_transform(world_from_patient_m).tolist()
        (folder / "patient_twin.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    return _publish(output, ".patient-twin-", write) / "patient_twin.yaml"
