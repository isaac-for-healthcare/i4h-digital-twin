# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quality metrics for one generated patient: CT, labels, twin meshes, centerlines, and bundle.

``evaluate`` measures; it does not judge. It returns JSON-serializable numbers, and
``make_report.py`` applies the pass/fail thresholds, so a finished cohort can be re-judged
without regenerating it.

All image metrics use the closest canonical (RAS) orientation of the generated NIfTI:
voxel axes point to the patient's right, anterior, and superior. Face names below
(``left``/``right``, ``posterior``/``anterior``, ``inferior``/``superior``) are patient
directions of the volume boundary that a structure touches.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from patient_digital_twin.anatomy import AnatomyCollection, canonical_name
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

#: Structures every chest-abdomen-pelvis twin should contain, whole (not cut by the scan boundary).
CORE = (
    "aorta", "heart", "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
    "lung_middle_lobe_right", "lung_lower_lobe_right", "liver", "spleen", "pancreas", "stomach",
    "kidney_left", "kidney_right", "inferior_vena_cava", "portal_vein_and_splenic_vein", "urinary_bladder",
)
LUNG_LOBES = tuple(name for name in CORE if name.startswith("lung_"))
#: Organ groups of NV-Generate's own CT check (upstream ``LDMSampler.label_int_dict``), as catalog names.
HU_GROUPS = {
    "liver": ("liver",), "spleen": ("spleen",), "pancreas": ("pancreas",), "kidney": ("kidney_left", "kidney_right"),
    "lung": LUNG_LOBES, "brain": ("brain",),
}
#: Adipose tissue attenuation window (HU), as used in CT body-composition studies.
ADIPOSE_HU = (-190.0, -30.0)
_FACES = ("left", "right", "posterior", "anterior", "inferior", "superior")


def label_ids(source_root: str | Path) -> dict[str, list[int]]:
    """Catalog name -> upstream label IDs (``configs/label_dict.json``), e.g. ``{"liver": [1]}``."""
    labels = json.loads((Path(source_root) / "configs/label_dict.json").read_text())
    result: dict[str, list[int]] = {}
    for raw, index in labels.items():
        result.setdefault(canonical_name(raw), []).append(int(index))
    return result


def canonical(image: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, np.ndarray]:
    """``(array, voxel size in mm)`` in the closest canonical RAS orientation."""
    image = nib.as_closest_canonical(image)
    return np.asanyarray(image.dataobj), np.asarray(image.header.get_zooms()[:3], dtype=float)


def _boundary_labels(labels: np.ndarray) -> dict[str, set[int]]:
    """Label IDs on each of the six faces of the volume."""
    planes = (labels[0], labels[-1], labels[:, 0], labels[:, -1], labels[:, :, 0], labels[:, :, -1])
    return {face: set(np.unique(plane).tolist()) for face, plane in zip(_FACES, planes)}


def label_metrics(labels: np.ndarray, spacing: np.ndarray, ids: Mapping[str, list[int]]) -> dict[str, Any]:
    """Present structures, volumes (mL), boundary contact, and fragmentation of the core organs."""
    voxel_ml = float(np.prod(spacing)) / 1000
    counts = np.bincount(labels.ravel().astype(np.int64))
    volume = {name: round(float(sum(counts[i] for i in index if i < len(counts))) * voxel_ml, 2)
              for name, index in ids.items() if name not in ("body",) and not name.startswith("dummy")}
    present = sorted(name for name, ml in volume.items() if ml > 0)
    faces = _boundary_labels(labels)
    touches = {name: [face for face in _FACES if set(ids[name]) & faces[face]] for name in CORE if name in ids}
    fragments = {}
    for name in CORE:
        mask = np.isin(labels, ids.get(name, []))
        if not mask.any():
            continue
        box = ndimage.find_objects(mask.astype(np.uint8))[0]
        components, count = ndimage.label(mask[box], structure=np.ones((3, 3, 3), bool))
        sizes = np.bincount(components.ravel())[1:]
        fragments[name] = {"components": int(count), "outside_largest": round(1 - sizes.max() / sizes.sum(), 5)}
    body = labels > 0
    return {
        "present": present,
        "volume_ml": {name: ml for name, ml in volume.items() if ml > 0},
        "lung_volume_ml": round(sum(volume.get(name, 0.0) for name in LUNG_LOBES), 1),
        "body_volume_l": round(float(body.sum()) * voxel_ml / 1000, 2),
        "core_missing": [name for name in CORE if volume.get(name, 0.0) == 0],
        "core_touching_boundary": {name: sides for name, sides in touches.items() if sides},
        "body_touching_boundary": [face for face in _FACES if (faces[face] - {0})],
        "fragments": fragments,
    }


def ct_metrics(ct: np.ndarray, labels: np.ndarray, spacing: np.ndarray, ids: Mapping[str, list[int]],
               statistics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """HU medians with NV-Generate's own outlier rule, adiposity, and L3 body-composition slice."""
    hu = {}
    for group, names in HU_GROUPS.items():
        index = [i for name in names for i in ids.get(name, [])]
        values = ct[np.isin(labels, index)]
        stats = statistics.get(group)
        if not len(values) or stats is None:
            continue
        low = min(stats["sigma_6_low"], stats["percentile_0_5"])
        high = max(stats["sigma_6_high"], stats["percentile_99_5"])
        median = float(np.median(values))
        hu[group] = {"median": round(median, 1), "low": round(low, 1), "high": round(high, 1),
                     "outlier": not low <= median <= high}
    body = labels > 0
    adipose = body & (ct >= ADIPOSE_HU[0]) & (ct <= ADIPOSE_HU[1])
    result: dict[str, Any] = {
        "hu": hu,
        "adipose_fraction": round(float(adipose.sum() / max(body.sum(), 1)), 4),
        "air_median_hu": round(float(np.median(ct[~body][::97])), 1) if (~body).any() else None,
    }
    l3 = np.isin(labels, ids.get("vertebrae_L3", []))
    if l3.any():
        k = int(round(np.argwhere(l3)[:, 2].mean()))
        slice_body, slice_ct, area = body[:, :, k], ct[:, :, k], float(spacing[0] * spacing[1]) / 100
        columns, rows = np.nonzero(slice_body.any(1))[0], np.nonzero(slice_body.any(0))[0]
        result["l3"] = {
            "slice": k,
            "body_area_cm2": round(float(slice_body.sum()) * area, 1),
            "adipose_area_cm2": round(float((slice_body & (slice_ct >= ADIPOSE_HU[0])
                                              & (slice_ct <= ADIPOSE_HU[1])).sum()) * area, 1),
            "width_mm": round(float(np.ptp(columns) + 1) * spacing[0], 1) if len(columns) else 0.0,
            "depth_mm": round(float(np.ptp(rows) + 1) * spacing[1], 1) if len(rows) else 0.0,
            "touches_side": bool(slice_body[0].any() or slice_body[-1].any()),
        }
    return result


def welded(vertices: np.ndarray, faces: np.ndarray) -> tuple[int, np.ndarray]:
    """``(vertex count, faces)`` after merging coincident vertices, as VTK's ``vtkCleanPolyData`` does
    before ``voxelize_mesh`` checks for a closed manifold."""
    unique, inverse = np.unique(np.asarray(vertices), axis=0, return_inverse=True)
    return len(unique), inverse.reshape(-1)[np.asarray(faces)]


def edge_counts(faces: np.ndarray, vertices: int) -> np.ndarray:
    """How many triangles share each undirected edge (1: boundary, 2: manifold, more: non-manifold)."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1).astype(np.int64)
    return np.unique(edges[:, 0] * vertices + edges[:, 1], return_counts=True)[1]


def mesh_metrics(anatomy: AnatomyCollection, volume_ml: Mapping[str, float]) -> dict[str, Any]:
    """Per-structure triangle counts, closed-manifold checks, components, and mesh-vs-voxel volume."""
    structures = {}
    for name, structure in anatomy.structures.items():
        vertices, faces = structure.mesh.vertices, structure.mesh.faces
        if vertices is None or faces is None:
            continue
        count, merged = welded(vertices, faces)
        counts = edge_counts(merged, count)
        graph = coo_matrix((np.ones(3 * len(merged)), (np.repeat(merged[:, 0], 3), merged.ravel())), shape=(count, count))
        triangles = vertices[faces]
        signed = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
        structures[name] = {
            "faces": int(len(faces)),
            "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()),
            "components": int(connected_components(graph, directed=False)[0]),
            "volume_ml": round(float(signed) * 1e6, 2),
        }
    closed = [n for n, s in structures.items() if s["boundary_edges"] == 0 and s["nonmanifold_edges"] == 0]
    errors = {n: round(s["volume_ml"] / volume_ml[n] - 1, 4) for n, s in structures.items()
              if n in CORE and volume_ml.get(n)}
    return {
        "structures": structures,
        "count": len(structures),
        "faces": int(sum(s["faces"] for s in structures.values())),
        "closed_manifold": len(closed),
        "not_closed_manifold": sorted(set(structures) - set(closed)),
        "core_volume_error": errors,
    }


def centerline_metrics(anatomy: AnatomyCollection) -> dict[str, Any]:
    """Graph size, endpoints, branch points, components, length (mm), and radius (mm) per vessel."""
    result = {}
    for name, structure in anatomy.structures.items():
        graph = structure.centerline
        if graph is None or not len(graph.edges):
            continue
        points, edges = np.asarray(graph.points) * 1000, np.asarray(graph.edges)
        degree = np.bincount(edges.ravel(), minlength=len(points))
        adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(points),) * 2)
        result[name] = {
            "points": int(len(points)),
            "endpoints": int((degree == 1).sum()),
            "branch_points": int((degree >= 3).sum()),
            "components": int(connected_components(adjacency, directed=False)[0]),
            "length_mm": round(float(np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1).sum()), 1),
            "radius_median_mm": round(float(np.median(graph.radii)) * 1000, 2),
            "radius_max_mm": round(float(np.max(graph.radii)) * 1000, 2),
        }
    return result


def bundle_metrics(manifest: Path) -> dict[str, Any]:
    """Re-read an exported bundle: manifest, arrays, and the USD stage's anatomy prims."""
    import yaml
    from pxr import Usd

    folder = manifest.parent
    data = yaml.safe_load(manifest.read_text())
    structures = data["anatomy"]["structures"]
    volume = np.load(folder / "volume.npy", mmap_mode="r")
    result: dict[str, Any] = {
        "schema_version": data.get("schema_version"),
        "structures": len(structures),
        "volume_shape": list(volume.shape),
        "size_mb": round(sum(f.stat().st_size for f in folder.iterdir()) / 2**20, 1),
    }
    if (folder / "vessel_mask.npy").is_file():
        result["vessel_mask_matches_volume"] = np.load(folder / "vessel_mask.npy", mmap_mode="r").shape == volume.shape
        result["centerline_points"] = int(np.load(folder / "centerline_points.npy").shape[0])
    stage = Usd.Stage.Open(str(folder / "patient_anatomy.usdc"))
    prims = {str(p.GetPath()) for p in stage.Traverse()}
    result["usd_prims_found"] = sum(entry["prim_path"] in prims for entry in structures.values())
    return result


def evaluate(*, ct_path: Path, labels_path: Path, anatomy: AnatomyCollection, manifest: Path | None,
             source_root: str | Path) -> dict[str, Any]:
    """All metrics for one patient (see the module docstring)."""
    ids = label_ids(source_root)
    statistics = json.loads((Path(source_root) / "configs/image_median_statistics_ct.json").read_text())
    labels, spacing = canonical(nib.load(str(labels_path)))
    ct, ct_spacing = canonical(nib.load(str(ct_path)))
    if ct.shape != labels.shape or not np.allclose(spacing, ct_spacing):
        raise ValueError("CT and labels must share their grid")
    labels = labels.astype(np.int32)
    metrics: dict[str, Any] = {"grid": {"shape": list(labels.shape), "spacing_mm": spacing.round(4).tolist()}}
    metrics["labels"] = label_metrics(labels, spacing, ids)
    metrics["ct"] = ct_metrics(np.asarray(ct, dtype=np.float32), labels, spacing, ids, statistics)
    metrics["mesh"] = mesh_metrics(anatomy, metrics["labels"]["volume_ml"])
    metrics["centerlines"] = centerline_metrics(anatomy)
    metrics["bundle"] = None if manifest is None else bundle_metrics(manifest)
    return metrics
