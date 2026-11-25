# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exporters: write a ``HumanBody`` as OpenUSD, as an i4h-workflows bundle, or as raw arrays.

Usually reached through ``HumanBody.export_to_usd`` and ``HumanBody.export_patient_twin``.

- ``export_to_usd``: one standalone stage with anatomy in its current pose, centerlines
  as custom prim attributes, and attached CT as native HU attributes. The stage uses the
  scan's units (meters without CT), Z-up, and default prim ``/HumanBody``.
- ``export_patient_twin``: a new schema-2 bundle directory: ``patient_twin.yaml``
  (manifest), ``patient_anatomy.usdc`` (anatomy in the scan's physical frame), and with
  CT ``volume.npy``/``volume.yaml``; ``vessel_names`` adds ``vessel_mask.npy`` and
  ``centerline_{points,edges,radii}.npy`` on the CT grid for navigation, plus one physics
  region per vessel: ``centerline_regions.npy`` and ``vessel_regions.yaml``.
- ``write_artifacts``: just the native CT and optional vessel mask/centerline/region arrays.
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
from .quality import DEFAULT_MAX_GAP_M, bridge_label_gaps, components, leaves_body_unchanged, repair
from .vessels import SOURCES, circulation, vessel_stiffness

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


def vessel_regions(names: Sequence[str]) -> dict[str, Any]:
    """The ``vessel_regions.yaml`` content for vessels whose region IDs are their positions in ``names``.

    It is an i4h-physics-simulation vessel region sidecar (schema 1): ``branches`` maps each
    region ID, as stored in ``centerline_regions.npy``, to its vessel, and ``young_modulus_kpa``
    gives each vessel's default wall stiffness from ``vessels.vessel_stiffness``.
    """
    return {
        "schema_version": 1,
        "default": "unassigned",
        "branches": dict(enumerate(names)),
        "young_modulus_kpa": {name: vessel_stiffness(name).young_modulus_kpa for name in names},
    }


def _write_scan(
    scan: ScanVolume,
    folder: Path,
    vessel_mask: ArrayLike | None = None,
    vessel_names: Sequence[str] = (),
    max_gap_m: float | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Native HU volume + YAML, plus an optional scan-grid vessel mask, its centerline, and vessel regions.

    ``vessel_mask`` is binary, or with ``vessel_names`` a label volume in which value ``i`` is
    ``vessel_names[i - 1]``. Labels add ``centerline_regions.npy`` (each edge's region ID, from the
    vessel containing the edge's first point) and ``vessel_regions.yaml``. ``max_gap_m`` bridges
    segmentation gaps between vessels of one circulation before skeletonizing
    (``quality.bridge_label_gaps``); ``vessel_mask.npy`` keeps the unbridged labels. Returns the
    artifact paths and a centerline report (bridges and remaining components).
    """
    scan.save(folder)
    paths = {"hu_volume": "volume.npy", "volume_metadata": "volume.yaml"}
    if vessel_mask is None:
        return paths, {}
    labels = np.asarray(vessel_mask)
    if labels.shape != scan.values.shape or not np.isin(labels, np.arange(max(len(vessel_names), 1) + 1)).all():
        raise ValueError("Vessel mask must match the scan grid and be binary, or hold labels 0..len(vessel_names)")
    report: dict[str, Any] = {}
    skeleton_labels = labels
    if max_gap_m:
        spacing = np.linalg.norm(scan.ijk_to_world[:3, :3], axis=0)[["ijk".index(c) for c in scan.array_axes]]
        groups = {name: i for i, name in enumerate(dict.fromkeys(circulation(n) for n in vessel_names))}
        group_of = [0] + [groups[circulation(n)] for n in vessel_names] if vessel_names else None
        skeleton_labels, bridges = bridge_label_gaps(labels, spacing, max_gap_m / scan.meters_per_unit,
                                                     circulation=group_of)
        report["centerline_bridges"] = [
            {**({"from": vessel_names[a - 1], "to": vessel_names[b - 1]} if vessel_names else {}),
             "gap_mm": round(gap * scan.meters_per_unit * 1000, 2)}
            for a, b, gap in bridges
        ]
    names = ("centerline_points", "centerline_edges", "centerline_radii")
    arrays: dict[str, np.ndarray] = {**dict(zip(names, native_centerline(skeleton_labels > 0, scan))),
                                     "vessel_mask": (labels > 0).astype(np.uint8)}
    if max_gap_m:
        report["centerline_components"] = components(arrays["centerline_edges"], len(arrays["centerline_points"]))
    point_labels = None
    if vessel_names:
        # Centerline points are skeleton voxel centers, so rounding recovers their voxels exactly.
        ijk = np.rint(transform_points(arrays["centerline_points"], np.linalg.inv(scan.ijk_to_world))).astype(int)
        point_labels = skeleton_labels.transpose([scan.array_axes.index(c) for c in "ijk"])[tuple(ijk.T)]
    if vessel_names and point_labels is not None:
        arrays["centerline_regions"] = (point_labels[arrays["centerline_edges"][:, 0]] - 1).astype(np.int32)
        sources = sorted({vessel_stiffness(n).source for n in vessel_names})
        header = "".join(f"# {key}: {SOURCES[key]}\n" for key in sources)
        (Path(folder) / "vessel_regions.yaml").write_text(
            "# Physics regions for centerline_regions.npy; Young's modulus sources:\n"
            + header + yaml.safe_dump(vessel_regions(vessel_names), sort_keys=False)
        )
        paths["vessel_regions"] = "vessel_regions.yaml"
    for name, values in arrays.items():
        np.save(Path(folder) / f"{name}.npy", values, allow_pickle=False)
        paths[name] = f"{name}.npy"
    return paths, report


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


def write_artifacts(
    scan: ScanVolume,
    output: str | Path,
    *,
    vessel_mask: ArrayLike | None = None,
    vessel_names: Sequence[str] = (),
    max_gap_m: float | None = DEFAULT_MAX_GAP_M,
) -> Path:
    """Write native HU ``volume.npy`` + ``volume.yaml`` and an optional vessel mask with its centerline.

    Args:
        scan: Source CT, e.g. ``scan_volume.from_nifti("ct.nii.gz")``.
        output: New directory (must not exist); written atomically.
        vessel_mask: Array with ``scan.values``' shape and axis order: binary, or a label volume
            (0 background, ``i`` for ``vessel_names[i - 1]``) when ``vessel_names`` is given.
        vessel_names: Vessel name per label; adds ``centerline_regions.npy`` and ``vessel_regions.yaml``.
        max_gap_m: Bridge vessel gaps up to this length before skeletonizing
            (``quality.bridge_label_gaps``); ``None`` skips.

    Returns:
        The output directory.
    """
    return _publish(output, ".ct-artifacts-",
                    lambda folder: _write_scan(scan, folder, vessel_mask, vessel_names, max_gap_m))


def _vessel_labels(body: HumanBody, scan: ScanVolume, names: Sequence[str]) -> NDArray[np.integer]:
    """Rasterize the requested vessel meshes onto the scan grid, exactly as the bundle USD places them.

    Returns a label volume in scan array order: 0 is background and ``i`` is ``names[i - 1]``.
    Where vessels overlap, the one listed first keeps the voxel. Each mesh goes through
    ``body.imaging.body_to_imaging @ local_to_body``, so a custom registration passed to
    ``attach_scan`` (or an edited structure placement) moves the navigation mask and
    centerline together with the exported vessel.
    """
    imaging = body.imaging
    if imaging is None:
        raise ValueError("Vessel masks require attached CT")
    scan_from_body = np.linalg.inv(scan.ijk_to_ras_m) @ imaging.body_to_imaging
    labels = np.zeros(scan.values_kji.shape, np.min_scalar_type(len(names)))
    for label, name in enumerate(names, 1):
        structure = body.anatomy.structures[name]
        vertices, faces = structure.mesh.vertices, structure.mesh.faces
        if vertices is None or faces is None:
            raise ValueError(f"Missing vessel mesh: {name}")
        try:
            inside = voxelize_mesh(
                transform_points(vertices, scan_from_body @ structure.local_to_body), faces,
                shape_zyx=labels.shape, spacing_zyx_m=(1.0, 1.0, 1.0), origin_xyz_m=(0.0, 0.0, 0.0),
            )
        except ValueError as exc:
            raise ValueError(f"Cannot rasterize vessel {name!r} onto the scan grid: {exc}") from exc
        labels[inside & (labels == 0)] = label
    if not labels.any():
        raise ValueError("Selected vessels have no foreground in the scan")
    return labels.transpose(["kji".index(c) for c in scan.array_axes])


@leaves_body_unchanged
def export_to_usd(body: HumanBody, path: str | Path, *, repair_body: bool = True) -> Path:
    """Write a standalone USD of ``body``: current pose, stored centerlines, and attached CT (native HU).

    The stage uses the scan's units (meters without CT), Z-up, and default prim ``/HumanBody``.
    An existing file is replaced only after the new layer is complete. Returns the resolved path.
    ``repair_body`` first repairs the meshes (``quality.repair``) for the export only; ``body`` is left as it was.

    Example:
        ``export_to_usd(body, "patient.usdc")`` (same as ``body.export_to_usd("patient.usdc")``)
    """
    if repair_body:
        repair(body)
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
        raise ImportError("USD export is optional; install with: uv pip install './patient-digital-twin[pipeline]'") from exc
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


@leaves_body_unchanged
def export_patient_twin(
    body: HumanBody,
    output: str | Path,
    *,
    patient_id: str | None = None,
    vessel_names: Sequence[str] = (),
    world_from_scan_m: ArrayLike | None = None,
    ct_exterior: bool = False,
    skin_opacity: float = 0.15,
    repair_body: bool = True,
    max_gap_m: float | None = DEFAULT_MAX_GAP_M,
) -> Path:
    """Write a schema-2 patient bundle directory and return its ``patient_twin.yaml`` path.

    Geometry uses the attached scan's physical frame and units (or the body/RAS frame in
    meters without CT). No CT canonicalization, mask cleanup, resampling, or simulator
    placement is performed. Structure centerlines stay on their USD prims.

    Args:
        output: New directory (must not exist).
        patient_id: Manifest ID; defaults to the CT source folder name or ``"geometry"``.
        vessel_names: Vessels to rasterize on the CT grid, with navigation centerline arrays.
            Each vessel becomes one physics region (``centerline_regions.npy``,
            ``vessel_regions.yaml``); overlapping voxels go to the vessel listed first.
        world_from_scan_m: Optional rigid simulator placement hint stored in the manifest. It maps
            scan-frame meters (scan coordinates times ``meters_per_unit``, in the scan's own
            RAS or LPS frame) to the simulator world, whatever ``spatial_unit`` is.
        ct_exterior: Add a CT-derived skin envelope at ``/HumanBody/Exterior/CT``.
        skin_opacity: Envelope opacity in [0, 1].
        repair_body: First repair the meshes (``quality.repair``; ``body`` itself is left unchanged) and bridge vessel
            gaps up to ``max_gap_m`` before skeletonizing the navigation centerline
            (``quality.bridge_label_gaps``); the manifest's ``quality`` section records both.
        max_gap_m: Largest vessel gap to bridge, in meters; ``None`` skips bridging.

    Example:
        ``export_patient_twin(body, "bundle", vessel_names=["aorta"], ct_exterior=True)``
    """
    if Path(output).expanduser().exists():
        raise FileExistsError(f"Use a new patient-twin output directory: {output}")
    scan = _scan(body)
    if ct_exterior and scan is None:
        raise ValueError("CT exterior requires attached CT")
    vessel_names = tuple(dict.fromkeys(vessel_names))
    for name in vessel_names:
        if name not in body.anatomy.structures or body.anatomy.structures[name].is_empty:
            raise ValueError(f"Missing or disabled vessel mesh: {name}")
    quality = repair(body) if repair_body else None
    gap = max_gap_m if repair_body else None
    units = scan.meters_per_unit if scan is not None else 1.0
    frame = scan.frame if scan is not None else "RAS" if body.anatomy.body_to_imaging is not None else "body"
    vessel_labels = _vessel_labels(body, scan, vessel_names) if scan is not None and vessel_names else None
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
            paths, centerline_report = _write_scan(scan, folder, vessel_labels, vessel_names, gap)
            artifacts.update(paths)
            if quality is not None:
                quality.update(centerline_report)
        else:
            folder.mkdir()
        prims = write_usd(body, folder / "patient_anatomy.usdc", units=units, scan=None,
                          placement=_scan_from_body(body, scan), exterior=exterior, skin_opacity=skin_opacity)
        structures = body.anatomy.structures
        manifest: dict[str, Any] = {
            "schema_version": 2,
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
        if world_from_scan_m is not None:
            manifest["transforms"]["world_from_scan_m"] = rigid_transform(world_from_scan_m).tolist()
        if quality is not None:
            manifest["quality"] = quality
        (folder / "patient_twin.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    return _publish(output, ".patient-twin-", write) / "patient_twin.yaml"
