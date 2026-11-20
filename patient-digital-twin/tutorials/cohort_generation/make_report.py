# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a generated cohort: quality-check verdicts, tables, statistics, and figures.

Reads ``cohort.json`` and every ``patients/<id>/record.json`` written by ``generate_cohort.py``
(and optionally a ``bmi_sweep.py`` folder), applies the pass/fail criteria in ``CRITERIA``, and
writes into ``--out``:

- ``results/summary.json``: every number quoted in REPORT.md;
- ``results/patients.csv``: one row per patient (characteristics, verdicts, key metrics);
- ``results/tables.md``: Markdown tables for the report;
- ``figures/*.png``: distributions, sample scans, rendered twins, the BMI sweep, fidelity plots.

Rendering twins needs VTK with off-screen OpenGL (EGL on a GPU machine, or OSMesa).

Example::

    python make_report.py --cohort cohort --sweep sweep --out .
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from cohort import Conditional, Derived  # noqa: E402
from presets import PRESETS  # noqa: E402
from qc import CORE, LUNG_LOBES, canonical  # noqa: E402
from scipy import stats  # noqa: E402

Record = dict[str, Any]

# ---------------------------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------------------------

#: Largest share of a core structure's voxels allowed outside its largest 26-connected component.
MAX_FRAGMENT_SHARE = 0.01
#: NV-Generate's merged label; its connectivity is reported separately (see REPORT.md).
PORTAL = "portal_vein_and_splenic_vein"
#: Smallest prostate label counted as an organ; smaller labels are spurious fragments.
PROSTATE_ML = 5.0


def _q(record: Record) -> Record:
    return record.get("qc") or {}


def _fragmented(record: Record) -> list[str]:
    return [n for n, f in _q(record)["labels"]["fragments"].items() if f["outside_largest"] > MAX_FRAGMENT_SHARE]


def _cut(record: Record) -> dict[str, list[str]]:
    """Core structures whose largest piece touches the scan boundary (``refine_completeness``)."""
    labels = _q(record)["labels"]
    return labels.get("core_cut", labels["core_touching_boundary"])


def _portal_connected(record: Record) -> bool:
    fragments = _q(record)["labels"]["fragments"].get(PORTAL)
    return fragments is not None and fragments["outside_largest"] <= MAX_FRAGMENT_SHARE


def _open_core_meshes(record: Record) -> list[str]:
    return [n for n in _q(record)["mesh"]["not_closed_manifold"] if n in CORE]


def _aorta_ok(record: Record) -> bool:
    graph = _q(record)["centerlines"].get("aorta")
    return graph is not None and graph["components"] == 1


def _bundle_ok(record: Record) -> bool:
    bundle = _q(record).get("bundle")
    return bool(bundle) and bundle["usd_prims_found"] == bundle["structures"] and bool(
        bundle.get("vessel_mask_matches_volume")) and bundle.get("centerline_points", 0) > 1


#: ``(key, description, test)``; every test except ``generated`` runs only on generated patients.
CRITERIA: list[tuple[str, str, Callable[[Record], bool]]] = [
    ("generated", "NV-Generate produced a CT and labels, and every pipeline step ran", lambda r: r["status"] == "ok"),
    ("ct_intensity", "Median HU of liver, spleen, pancreas, kidneys, lungs, and any brain label within NV-Generate's "
     "own acceptance ranges",
     lambda r: not any(v["outlier"] for v in _q(r)["ct"]["hu"].values())),
    ("core_present", f"All {len(CORE)} core structures labeled", lambda r: not _q(r)["labels"]["core_missing"]),
    ("core_complete", "No core structure's largest piece touches the scan boundary (none cut off)",
     lambda r: not _cut(r)),
    ("core_connected", f"Each core structure except the merged portal/splenic vein label has at least "
     f"{1 - MAX_FRAGMENT_SHARE:.0%} of its voxels in one 26-connected piece",
     lambda r: not set(_fragmented(r)) - {PORTAL}),
    ("core_meshes_closed", "Every core mesh is a closed 2-manifold (no boundary or non-manifold edges)",
     lambda r: not _open_core_meshes(r)),
    ("aorta_centerline", "Aorta centerline extracted as one connected graph", _aorta_ok),
    ("bundle", "Bundle exported and re-read: all USD anatomy prims, CT-grid vessel mask, navigation centerline", _bundle_ok),
]
#: Reported but not required: wide patients may exceed the 486 mm field of view, as in clinical
#: CT, and the merged portal/splenic vein label is rarely one piece (see REPORT.md).
FINDINGS: list[tuple[str, str, Callable[[Record], bool]]] = [
    ("body_in_fov", "Body not cut by the left, right, anterior, or posterior scan boundary",
     lambda r: not set(_q(r)["labels"]["body_touching_boundary"]) - {"inferior", "superior"}),
    ("portal_vein_connected", f"Portal/splenic vein label has at least {1 - MAX_FRAGMENT_SHARE:.0%} of its voxels "
     "in one 26-connected piece", _portal_connected),
]


def ready_with_mesh_fix(record: Record) -> bool:
    """Estimated ``twin_ready`` if marching cubes' back-to-back triangle pairs were removed.

    ``mesh_pair_analysis`` shows the removal closes every non-closed mesh, so ``core_meshes_closed``
    passes, as do aorta centerline and bundle failures caused by a non-closed aorta mesh.
    """
    result = verdicts(record)
    if not result["generated"]:
        return False
    result["core_meshes_closed"] = True
    aorta_open = "aorta" in _q(record)["mesh"]["not_closed_manifold"]
    if aorta_open and "closed manifold" in _q(record).get("centerline_failures", {}).get("aorta", ""):
        result["aorta_centerline"] = True
    if aorta_open and "closed manifold" in record.get("bundle_error", ""):
        result["bundle"] = True
    return all(result[key] for key, _, _ in CRITERIA)


def verdicts(record: Record) -> dict[str, bool]:
    """Pass/fail per criterion and finding, plus ``twin_ready`` (every criterion)."""
    result = {key: False for key, _, _ in CRITERIA + FINDINGS}
    result["generated"] = record["status"] == "ok"
    if result["generated"]:
        for key, _, test in CRITERIA[1:] + FINDINGS:
            result[key] = bool(test(record))
    result["twin_ready"] = all(result[key] for key, _, _ in CRITERIA)
    return result


# ---------------------------------------------------------------------------------------------
# Loading and flat tables
# ---------------------------------------------------------------------------------------------


_FACES = {"left": (0, 0), "right": (0, -1), "posterior": (1, 0), "anterior": (1, -1), "inferior": (2, 0), "superior": (2, -1)}


def refine_completeness(folder: Path, records: Sequence[Record], ids: Mapping[str, list[int]]) -> None:
    """Add ``labels.core_cut``: boundary contacts of each structure's largest piece (a real cut).

    ``qc.label_metrics`` flags any voxel on the boundary, so a stray fragment far from the organ
    also counts; ``core_complete`` judges only the largest 26-connected piece.
    """
    from scipy import ndimage

    for record in records:
        if record["status"] != "ok":
            continue
        labels_qc = record["qc"]["labels"]
        labels_qc["core_cut"] = {}
        if not labels_qc["core_touching_boundary"]:
            continue
        labels, _ = canonical(nib.load(str(folder / "patients" / record["id"] / "labels.nii.gz")))
        for name, sides in labels_qc["core_touching_boundary"].items():
            pieces, _ = ndimage.label(np.isin(labels, ids[name]), structure=np.ones((3, 3, 3), bool))
            largest = pieces == np.argmax(np.bincount(pieces.ravel())[1:]) + 1
            if cut := [side for side in sides if np.take(largest, _FACES[side][1], axis=_FACES[side][0]).any()]:
                labels_qc["core_cut"][name] = cut


def load(folder: Path) -> tuple[Record, list[Record]]:
    """``(cohort.json or sweep.json, records in patient order)``; missing records are skipped."""
    manifest = folder / ("cohort.json" if (folder / "cohort.json").is_file() else "sweep.json")
    cohort = json.loads(manifest.read_text())
    records = [json.loads(p.read_text()) for p in sorted((folder / "patients").glob("*/record.json"))]
    return cohort, records


def flat(record: Record) -> dict[str, Any]:
    """One CSV row: characteristics, verdicts, and key metrics."""
    row: dict[str, Any] = {k: v for k, v in record["patient"].items()}
    row.update(verdicts(record))
    q = _q(record)
    if record["status"] == "ok":
        labels, ct = q["labels"], q["ct"]
        volume = labels["volume_ml"]
        row.update({
            "body_volume_l": labels["body_volume_l"], "adipose_fraction": ct["adipose_fraction"],
            "l3_body_area_cm2": (ct.get("l3") or {}).get("body_area_cm2"),
            "l3_adipose_area_cm2": (ct.get("l3") or {}).get("adipose_area_cm2"),
            "l3_width_mm": (ct.get("l3") or {}).get("width_mm"),
            "lung_volume_ml": labels["lung_volume_ml"],
            "kidney_volume_ml": round(volume.get("kidney_left", 0) + volume.get("kidney_right", 0), 1),
            **{f"{n}_volume_ml": volume.get(n) for n in ("liver", "spleen", "heart", "pancreas", "prostate")},
            "aorta_diameter_mm": round(2 * q["centerlines"]["aorta"]["radius_median_mm"], 1) if "aorta" in q["centerlines"] else None,
            "aorta_centerline_mm": q["centerlines"].get("aorta", {}).get("length_mm"),
            "structures": len(labels["present"]), "mesh_faces": q["mesh"]["faces"],
            "fragmented": ";".join(_fragmented(record)), "open_core_meshes": ";".join(_open_core_meshes(record)),
            "cut_core": ";".join(f"{n}:{'/'.join(s)}" for n, s in _cut(record).items()),
            "centerline_failures": ";".join(q.get("centerline_failures", {})),
            "seconds": record["timings_s"].get("total"),
        })
    else:
        row["error"] = record.get("error")
    return row


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def markdown(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def _mean_sd(values: Sequence[float], digits: int = 1) -> str:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return "n/a"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 else f"{values[0]:.{digits}f}"


def spearman(x: Sequence[float], y: Sequence[float], seed: int = 0, resamples: int = 2000) -> dict[str, float]:
    """Spearman rho with a percentile bootstrap 95% CI and the two-sided p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(resamples):
        index = rng.integers(0, len(x), len(x))
        boot.append(stats.spearmanr(x[index], y[index])[0])
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return {"rho": round(float(rho), 3), "ci_low": round(float(low), 3), "ci_high": round(float(high), 3),
            "p": float(p), "n": int(len(x))}


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------

#: Render colors (RGB 0-1) and opacities for the twin figures.
PALETTE = {
    "skin": ((0.93, 0.80, 0.70), 0.12), "lung": ((0.55, 0.75, 0.95), 0.35), "heart": ((0.70, 0.12, 0.15), 1.0),
    "aorta": ((0.95, 0.10, 0.10), 1.0), "vein": ((0.20, 0.35, 0.85), 1.0), "liver": ((0.55, 0.27, 0.15), 1.0),
    "spleen": ((0.50, 0.20, 0.55), 1.0), "kidney": ((0.85, 0.55, 0.25), 1.0), "pancreas": ((0.95, 0.80, 0.35), 1.0),
    "stomach": ((0.90, 0.65, 0.65), 0.9), "bladder": ((0.45, 0.80, 0.45), 0.9), "bone": ((0.92, 0.90, 0.82), 0.55),
}
_GROUP = {"aorta": "aorta", "iliac_artery_left": "aorta", "iliac_artery_right": "aorta", "brachiocephalic_trunk": "aorta",
          "common_carotid_artery_left": "aorta", "common_carotid_artery_right": "aorta", "subclavian_artery_left": "aorta",
          "subclavian_artery_right": "aorta", "inferior_vena_cava": "vein", "superior_vena_cava": "vein",
          "portal_vein_and_splenic_vein": "vein", "iliac_vena_left": "vein", "iliac_vena_right": "vein",
          "heart": "heart", "liver": "liver", "spleen": "spleen", "kidney_left": "kidney", "kidney_right": "kidney",
          "pancreas": "pancreas", "stomach": "stomach", "urinary_bladder": "bladder", **{n: "lung" for n in LUNG_LOBES}}


def _bone(name: str) -> bool:
    return name.startswith(("rib_", "vertebrae_", "hip_", "sacrum", "sternum", "clavicula_", "scapula_", "femur_",
                            "humerus_"))


def render_twin(usd_path: Path, size: tuple[int, int] = (600, 800), half_height_mm: float = 345,
                view: str = "anterior", skin: bool = True) -> np.ndarray:
    """Off-screen VTK render of a bundle's ``patient_anatomy.usdc`` as an RGB array.

    A parallel projection at a fixed scale (``half_height_mm`` per half image height),
    centered on the scan, keeps every render at the same millimeters per pixel, so body
    sizes compare across patients.
    """
    import vtk
    from pxr import Usd, UsdGeom
    from vtk.util.numpy_support import (
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
        vtk_to_numpy,
    )

    stage = Usd.Stage.Open(str(usd_path))
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.SetUseDepthPeeling(1)
    renderer.SetMaximumNumberOfPeels(8)
    bounds = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = prim.GetCustomDataByKey("anatomy:name") or ("skin" if "/Exterior/" in str(prim.GetPath()) else None)
        group = "skin" if name == "skin" else _GROUP.get(name, "bone" if name and _bone(name) else None)
        if group is None or (group == "skin" and not skin):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float)
        points = points @ matrix[:3, :3] + matrix[3, :3]  # USD row-vector convention.
        bounds.append((points.min(0), points.max(0)))
        polydata = vtk.vtkPolyData()
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points, deep=True))
        cells = vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True),
                      numpy_to_vtkIdTypeArray(faces.ravel(), deep=True))
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(cells)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.SetFeatureAngle(80)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        color, opacity = PALETTE[group]
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetSpecular(0.15)
        renderer.AddActor(actor)
    low = np.min([b[0] for b in bounds], 0)
    high = np.max([b[1] for b in bounds], 0)
    center = (low + high) / 2
    camera = renderer.GetActiveCamera()
    direction = {"anterior": (0, 1, 0), "left": (-1, 0, 0)}[view]  # RAS: +y anterior, -x patient's left.
    camera.ParallelProjectionOn()
    camera.SetParallelScale(half_height_mm)
    camera.SetFocalPoint(*center)
    camera.SetPosition(*(center + 2000 * np.asarray(direction)))
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCameraClippingRange()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetAlphaBitPlanes(1)
    window.SetMultiSamples(0)
    window.AddRenderer(renderer)
    window.SetSize(*size)
    window.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    grab.Update()
    image = grab.GetOutput()
    width, height, _ = image.GetDimensions()
    rgb = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(height, width, -1)[::-1, :, :3]
    window.Finalize()
    return np.ascontiguousarray(rgb)


def _crop_together(images: Sequence[np.ndarray], margin: int = 4) -> list[np.ndarray]:
    """Crop equally sized renders to the union of their non-white content (keeps one scale)."""
    content = np.argwhere(np.any([(image < 250).any(-1) for image in images], axis=0))
    (top, left), (bottom, right) = content.min(0), content.max(0) + 1
    return [image[max(top - margin, 0):bottom + margin, max(left - margin, 0):right + margin] for image in images]


def _axial(ct: np.ndarray, labels: np.ndarray, ids_l3: Sequence[int]) -> np.ndarray:
    k = int(round(np.argwhere(np.isin(labels, ids_l3))[:, 2].mean())) if np.isin(labels, ids_l3).any() else ct.shape[2] // 2
    return ct[:, :, k].T[::-1, ::-1]  # Anterior up, patient's right on the image left.


def _coronal(ct: np.ndarray, labels: np.ndarray, aorta_ids: Sequence[int]) -> np.ndarray:
    aorta = np.isin(labels, aorta_ids)
    j = int(round(np.argwhere(aorta)[:, 1].mean())) if aorta.any() else ct.shape[1] // 2
    return ct[:, j, :].T[::-1, ::-1]  # Superior up, patient's right on the image left.


def _title(patient: Mapping[str, Any]) -> str:
    return f"{patient['id']}: {patient['sex']}, {patient['age']:.0f} y, BMI {patient['bmi']:.1f}"


def figure_scans(folder: Path, patients: Sequence[Mapping[str, Any]], ids: Mapping[str, list[int]], path: Path) -> None:
    """Axial slice at L3 and coronal slice through the aorta for each patient."""
    fig, axes = plt.subplots(2, len(patients), figsize=(2.6 * len(patients), 6.2), gridspec_kw={"height_ratios": [1, 1.32]},
                             squeeze=False)
    for column, patient in enumerate(patients):
        ct, spacing = canonical(nib.load(str(folder / "patients" / patient["id"] / "ct.nii.gz")))
        labels, _ = canonical(nib.load(str(folder / "patients" / patient["id"] / "labels.nii.gz")))
        ax = axes[0, column]
        ax.imshow(_axial(ct, labels, ids["vertebrae_L3"]), cmap="gray", vmin=-160, vmax=240, aspect=spacing[1] / spacing[0])
        ax.set_title(_title(patient), fontsize=8)
        ax = axes[1, column]
        ax.imshow(_coronal(ct, labels, ids["aorta"]), cmap="gray", vmin=-1000, vmax=500, aspect=spacing[2] / spacing[0])
        for ax in axes[:, column]:
            ax.axis("off")
    axes[0, 0].text(-0.08, 0.5, "axial at L3\n(W400 L40)", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    axes[1, 0].text(-0.08, 0.5, "coronal through aorta\n(W1500 L-250)", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def figure_twins(folder: Path, patients: Sequence[Mapping[str, Any]], path: Path, columns: int | None = None,
                 title: Callable[[Mapping[str, Any]], str] | None = None) -> None:
    """Rendered bundle anatomy (anterior view, one scale) per patient, ``columns`` per row, with a legend."""
    columns = columns or len(patients)
    rows = math.ceil(len(patients) / columns)
    usds = [folder / "patients" / p["id"] / "twin/patient_anatomy.usdc" for p in patients]
    rendered = [render_twin(usd) for usd in usds if usd.is_file()]
    images = iter(_crop_together(rendered) if rendered else [])
    fig, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 3.4 * rows + 0.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, patient, usd in zip(axes.ravel(), patients, usds):
        if usd.is_file():
            ax.imshow(next(images))
        else:
            ax.text(0.5, 0.5, "no bundle:\nexport failed", ha="center", va="center", fontsize=8, transform=ax.transAxes)
        ax.set_title((title or _title)(patient), fontsize=8)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, alpha=max(a, 0.5), markersize=8, label=g)
               for g, (c, a) in PALETTE.items()]
    fig.legend(handles=handles, loc="lower center", ncol=(len(handles) + 1) // 2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.5 / (3.4 * rows + 0.5) + 0.03, 1, 1))
    fig.savefig(path, dpi=80, pil_kwargs={"quality": 88} if path.suffix == ".jpg" else None)
    plt.close(fig)


def figure_twin_detail(folder: Path, patient: Mapping[str, Any], path: Path) -> None:
    """One twin from the front and the left, without the skin envelope."""
    usd = folder / "patients" / patient["id"] / "twin/patient_anatomy.usdc"
    images = _crop_together([render_twin(usd, size=(900, 1100), half_height_mm=330, view=view, skin=False)
                             for view in ("anterior", "left")])
    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    for ax, image, view in zip(axes, images, ("anterior view", "left lateral view")):
        ax.imshow(image)
        ax.set_title(view, fontsize=9)
        ax.axis("off")
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, alpha=max(a, 0.5), markersize=8, label=g)
               for g, (c, a) in PALETTE.items() if g != "skin"]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.suptitle(_title(patient), fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=90, pil_kwargs={"quality": 88} if path.suffix == ".jpg" else None)
    plt.close(fig)


def figure_distributions(patients: Sequence[Mapping[str, Any]], spec: Any, path: Path) -> None:
    """Sampled characteristics against their target densities (by sex where the target is sex-specific)."""
    colors = {"M": "#3366cc", "F": "#dc3912"}
    names = [n for n in ("age", "height", "bmi", "weight") if n in spec.characteristics]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(3.3 * (len(names) + 1), 3.0))
    counts = {s: sum(p["sex"] == s for p in patients) for s in ("M", "F")}
    axes[0].bar(list(counts), list(counts.values()), color=[colors[s] for s in counts])
    expected = getattr(spec.characteristics["sex"], "p", None)
    if expected is not None:
        n = len(patients)
        low, high = stats.binom.ppf([0.025, 0.975], n, expected)
        axes[0].fill_between([-0.45, 0.45], low, high, color="k", alpha=0.2, label=f"95% range of Binomial({n}, {expected:g})")
        axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axes[0].set_title("sex (count)")
    for ax, name in zip(axes[1:], names):
        distribution = spec.characteristics[name]
        values = np.array([p[name] for p in patients], float)
        grid = np.linspace(values.min() - 2, values.max() + 2, 300)
        by_sex = isinstance(distribution, Conditional) and distribution.on == "sex"
        if by_sex or isinstance(distribution, Derived):
            for sex in ("M", "F"):
                subset = np.array([p[name] for p in patients if p["sex"] == sex], float)
                ax.hist(subset, bins=24, density=True, alpha=0.35, color=colors[sex], label=f"{sex} sampled")
                density = distribution.pdf(grid, sex) if by_sex else None
                if density is not None:
                    ax.plot(grid, density, color=colors[sex], lw=1.4, label=f"{sex} target")
        else:
            ax.hist(values, bins=24, density=True, alpha=0.4, color="0.55", label="sampled")
            density = distribution.pdf(grid)
            if density is not None:
                ax.plot(grid, density, color="k", lw=1.4, label="target")
        ax.set_title({"age": "age (years)", "height": "height (cm)", "bmi": "BMI (kg/m²)", "weight": "weight (kg)"}[name])
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def figure_sweep(folder: Path, records: Sequence[Record], ids: Mapping[str, list[int]], path: Path) -> None:
    """The BMI sweep: L3 slices per (seed, sex) row and BMI column, plus body measures against BMI."""
    rows = sorted({(r["patient"]["seed"], r["patient"]["sex"]) for r in records})
    bmis = sorted({r["patient"]["bmi"] for r in records})
    by_key = {(r["patient"]["seed"], r["patient"]["sex"], r["patient"]["bmi"]): r for r in records}
    shown = rows[:4]
    fig = plt.figure(figsize=(2.1 * len(bmis) + 4.2, 2.0 * len(shown) + 0.4))
    grid = fig.add_gridspec(len(shown), len(bmis) + 2)
    for row, (seed, sex) in enumerate(shown):
        for column, bmi in enumerate(bmis):
            record = by_key.get((seed, sex, bmi))
            ax = fig.add_subplot(grid[row, column])
            ax.axis("off")
            if record is None or record["status"] != "ok":
                continue
            ct, spacing = canonical(nib.load(str(folder / "patients" / record["id"] / "ct.nii.gz")))
            labels, _ = canonical(nib.load(str(folder / "patients" / record["id"] / "labels.nii.gz")))
            ax.imshow(_axial(ct, labels, ids["vertebrae_L3"]), cmap="gray", vmin=-160, vmax=240)
            if row == 0:
                ax.set_title(f"BMI {bmi:g}", fontsize=9)
            if column == 0:
                ax.text(-0.05, 0.5, f"seed {seed}, {sex}", transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=8)
    for index, (metric, label) in enumerate((("l3_adipose_area_cm2", "L3 adipose area (cm²)"), ("body_volume_l", "body volume in scan (L)"))):
        ax = fig.add_subplot(grid[:, len(bmis) + index])
        for seed, sex in rows:
            series = [(bmi, flat(by_key[(seed, sex, bmi)]).get(metric)) for bmi in bmis if (seed, sex, bmi) in by_key
                      and by_key[(seed, sex, bmi)]["status"] == "ok"]
            ax.plot(*zip(*series), marker="o", ms=3, color={"M": "#3366cc", "F": "#dc3912"}[sex], alpha=0.7)
        ax.set_xlabel("conditioned BMI (kg/m²)")
        ax.set_title(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def figure_fidelity(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Conditioning fidelity: BMI vs body measures, sex vs organ volumes, age vs aortic diameter."""
    ok = [r for r in rows if r["generated"]]
    colors = {"M": "#3366cc", "F": "#dc3912"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, metric, label in ((axes[0], "l3_adipose_area_cm2", "L3 adipose area (cm²)"),
                              (axes[1], "body_volume_l", "body volume in scan (L)")):
        for sex in ("M", "F"):
            subset = [r for r in ok if r["sex"] == sex and r.get(metric) is not None]
            ax.scatter([r["bmi"] for r in subset], [r[metric] for r in subset], s=6, alpha=0.5, color=colors[sex], label=sex)
        result = spearman([r["bmi"] for r in ok], [r.get(metric) for r in ok])
        ax.set_title(f"{label}\nSpearman ρ = {result['rho']:.2f}", fontsize=9)
        ax.set_xlabel("conditioned BMI (kg/m²)")
        ax.legend(fontsize=7)
    organs = ("liver_volume_ml", "kidney_volume_ml", "heart_volume_ml", "lung_volume_ml")
    labels = ("liver", "kidneys", "heart", "lungs/4")
    positions = np.arange(len(organs))
    for offset, sex in ((-0.18, "M"), (0.18, "F")):
        data = [[r[o] / (4 if o == "lung_volume_ml" else 1) for r in ok if r["sex"] == sex and r.get(o)] for o in organs]
        parts = axes[2].boxplot(data, positions=positions + offset, widths=0.3, patch_artist=True, showfliers=False)
        for box in parts["boxes"]:
            box.set_facecolor(colors[sex])
            box.set_alpha(0.5)
        for position, organ in zip(positions + offset, organs):
            if organ in REFERENCES:  # CT reference mean or median (REPORT.md, references).
                _, male, female, _ = REFERENCES[organ]
                value = (male if sex == "M" else female) / (4 if organ == "lung_volume_ml" else 1)
                axes[2].plot([position - 0.15, position + 0.15], [value] * 2, color="k", lw=2,
                             label="CT reference" if sex == "M" and position == offset else None)
    axes[2].legend(fontsize=7, loc="upper right")
    axes[2].set_xticks(positions, labels)
    axes[2].set_ylabel("volume (mL)")
    axes[2].set_title("organ volume by sex (M blue, F red)", fontsize=9)
    for sex in ("M", "F"):
        subset = [r for r in ok if r["sex"] == sex and r.get("aorta_diameter_mm")]
        axes[3].scatter([r["age"] for r in subset], [r["aorta_diameter_mm"] for r in subset], s=6, alpha=0.5, color=colors[sex], label=sex)
    ages = np.linspace(20, 87, 50)
    for sex, (base, slope) in {"M": (19.3, 0.13), "F": (16.7, 0.09)}.items():  # Rogers 2013, infrarenal.
        axes[3].plot(ages, base + slope * (ages - 51), "--", color=colors[sex], lw=1.2,
                     label=f"{sex}, Framingham infrarenal")
    result = spearman([r["age"] for r in ok], [r.get("aorta_diameter_mm") for r in ok])
    axes[3].set_title(f"aorta median diameter (centerline)\nSpearman ρ with age = {result['rho']:.2f}", fontsize=9)
    axes[3].set_xlabel("conditioned age (years)")
    axes[3].set_ylabel("mm")
    axes[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------


def _pick(rows: Sequence[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    """Twin-ready patients spread over BMI, alternating sex, for the sample figures."""
    ready = sorted((r for r in rows if r["twin_ready"]), key=lambda r: r["bmi"]) or sorted(
        (r for r in rows if r["generated"]), key=lambda r: r["bmi"])
    chosen: list[Mapping[str, Any]] = []
    for quantile in np.linspace(0.02, 0.98, count):
        target = np.quantile([r["bmi"] for r in ready], quantile)
        sex = "MF"[len(chosen) % 2]
        candidates = [r for r in ready if r not in chosen and r["sex"] == sex] or [r for r in ready if r not in chosen]
        if not candidates:
            break
        chosen.append(min(candidates, key=lambda r: abs(r["bmi"] - target)))
    return chosen


def summarize(cohort: Record, records: Sequence[Record], rows: Sequence[Mapping[str, Any]]) -> Record:
    """Numbers for REPORT.md."""
    ok = [r for r in records if r["status"] == "ok"]
    generated = [r for r in rows if r["generated"]]
    n = len(rows)
    summary: Record = {"patients": n, "generated": len(ok)}
    summary["criteria"] = {key: {"description": text, "passed": sum(r[key] for r in rows)}
                           for key, text, _ in CRITERIA}
    summary["criteria"]["twin_ready"] = {"description": "All criteria", "passed": sum(r["twin_ready"] for r in rows)}
    summary["findings"] = {key: {"description": text, "passed": sum(r[key] for r in rows)} for key, text, _ in FINDINGS}
    summary["twin_ready_with_mesh_fix"] = sum(ready_with_mesh_fix(r) for r in records)
    summary["aorta_centerline_components"] = dict(sorted(Counter(
        r["qc"]["centerlines"]["aorta"]["components"] for r in ok if "aorta" in r["qc"]["centerlines"]).items()))
    tally: dict[str, dict[str, int]] = {}
    for record in ok:
        labels = record["qc"]["labels"]
        for name in labels["core_missing"]:
            tally.setdefault(name, {}).setdefault("missing", 0)
            tally[name]["missing"] += 1
        for name in labels["core_touching_boundary"]:
            kind = "cut" if name in _cut(record) else "stray_fragment_at_boundary"
            tally.setdefault(name, {}).setdefault(kind, 0)
            tally[name][kind] += 1
        for name in _fragmented(record):
            tally.setdefault(name, {}).setdefault("fragmented", 0)
            tally[name]["fragmented"] += 1
        for name in _open_core_meshes(record):
            tally.setdefault(name, {}).setdefault("open_mesh", 0)
            tally[name]["open_mesh"] += 1
    summary["core_failures"] = dict(sorted(tally.items()))
    centerline_failures: dict[str, int] = {}
    for record in ok:
        for name in record["qc"].get("centerline_failures", {}):
            centerline_failures[name] = centerline_failures.get(name, 0) + 1
    summary["centerline_failures"] = dict(sorted(centerline_failures.items(), key=lambda item: -item[1]))
    summary["all_meshes_closed"] = sum(not r["qc"]["mesh"]["not_closed_manifold"] for r in ok)
    summary["mesh_open_structures"] = dict(sorted(
        {n: sum(n in r["qc"]["mesh"]["not_closed_manifold"] for r in ok) for r in ok
         for n in r["qc"]["mesh"]["not_closed_manifold"]}.items(), key=lambda item: -item[1])[:12])
    errors = [abs(e) for r in ok for e in r["qc"]["mesh"]["core_volume_error"].values()]
    summary["mesh_volume_error"] = {"median": round(float(np.median(errors)), 4), "max": round(float(np.max(errors)), 4)} if errors else None
    summary["hu_outliers"] = {g: sum(r["qc"]["ct"]["hu"].get(g, {}).get("outlier", False) for r in ok) for g in
                              ("liver", "spleen", "pancreas", "kidney", "lung")}
    summary["fidelity"] = {
        f"bmi_vs_{m}": spearman([r["bmi"] for r in generated], [r.get(m) for r in generated])
        for m in ("body_volume_l", "l3_body_area_cm2", "l3_adipose_area_cm2", "adipose_fraction", "l3_width_mm")
    }
    for sex in ("M", "F"):
        subset = [r for r in generated if r["sex"] == sex]
        summary["fidelity"][f"bmi_vs_l3_adipose_area_cm2_{sex}"] = spearman([r["bmi"] for r in subset],
                                                                            [r.get("l3_adipose_area_cm2") for r in subset])
    summary["fidelity"]["age_vs_aorta_diameter_mm"] = spearman([r["age"] for r in generated], [r.get("aorta_diameter_mm") for r in generated])
    organ_keys = ("liver_volume_ml", "spleen_volume_ml", "kidney_volume_ml", "heart_volume_ml", "lung_volume_ml",
                  "pancreas_volume_ml", "aorta_diameter_mm", "l3_adipose_area_cm2", "body_volume_l")
    summary["by_sex"] = {}
    for key in organ_keys:
        male = [r[key] for r in generated if r["sex"] == "M" and r.get(key)]
        female = [r[key] for r in generated if r["sex"] == "F" and r.get(key)]
        test = stats.mannwhitneyu(male, female) if male and female else None
        summary["by_sex"][key] = {"M": _mean_sd(male), "F": _mean_sd(female), "ratio_M_F": round(np.mean(male) / np.mean(female), 3)
                                  if male and female else None, "p": None if test is None else float(test.pvalue)}
    summary["prostate"] = {}  # Organ-sized (>= 5 mL) versus small spurious label fragments.
    for sex in ("M", "F"):
        volumes = [r.get("prostate_volume_ml") or 0.0 for r in generated if r["sex"] == sex]
        organ = [v for v in volumes if v >= PROSTATE_ML]
        summary["prostate"][sex] = {"patients": len(volumes), "organ": len(organ),
                                    "median_ml": round(float(np.median(organ)), 1) if organ else None,
                                    "fragments_only": sum(0 < v < PROSTATE_ML for v in volumes),
                                    "largest_fragment_ml": max([v for v in volumes if v < PROSTATE_ML], default=0.0)}
    bmi_classes = ((0, 18.5), (18.5, 25), (25, 30), (30, 35), (35, 40), (40, 100))
    summary["bmi_classes"] = []
    for low, high in bmi_classes:
        subset = [r for r in generated if low <= r["bmi"] < high]
        if subset:
            summary["bmi_classes"].append({
                "class": f"{low:g}–{high:g}" if high < 100 else f"≥ {low:g}", "n": len(subset),
                "body_volume_l": round(float(np.median([r["body_volume_l"] for r in subset])), 1),
                "l3_adipose_area_cm2": round(float(np.nanmedian([r.get("l3_adipose_area_cm2") or np.nan for r in subset])), 0),
                "body_in_fov": f"{sum(r['body_in_fov'] for r in subset)}/{len(subset)}",
                "twin_ready": f"{sum(r['twin_ready'] for r in subset)}/{len(subset)}",
            })
    seconds = [r["timings_s"].get("total") for r in records if r["timings_s"].get("total")]
    stages = ("generate_and_mesh", "centerlines", "bundle", "qc", "total")
    summary["timing_s"] = {stage: round(float(np.median([r["timings_s"][stage] for r in ok if stage in r["timings_s"]])), 1)
                           for stage in stages if ok}
    summary["timing_s"]["sum_hours"] = round(sum(seconds) / 3600, 2)
    summary["sizes"] = {"bundle_mb_median": round(float(np.median([r["qc"]["bundle"]["size_mb"] for r in ok if r["qc"].get("bundle")])), 1)
                        if ok else None,
                        "structures_median": int(np.median([len(r["qc"]["labels"]["present"]) for r in ok])) if ok else None,
                        "mesh_faces_median": int(np.median([r["qc"]["mesh"]["faces"] for r in ok])) if ok else None}
    # Presence of every structure, by sex: sex-specific anatomy is a direct check of the sex condition.
    presence: dict[str, dict[str, int]] = {}
    for record in ok:
        for name in record["qc"]["labels"]["present"]:
            presence.setdefault(name, {"M": 0, "F": 0})[record["patient"]["sex"]] += 1
    summary["presence_by_sex"] = {"patients": {s: sum(r["patient"]["sex"] == s for r in ok) for s in ("M", "F")},
                                  "structures": dict(sorted(presence.items()))}
    summary["hu_median"] = {group: round(float(np.median([r["qc"]["ct"]["hu"][group]["median"] for r in ok
                                                          if group in r["qc"]["ct"]["hu"]])), 1)
                            for group in ("liver", "spleen", "pancreas", "kidney", "lung") if ok}
    for sex in ("M", "F"):
        subset = [r for r in generated if r["sex"] == sex]
        summary["fidelity"][f"age_vs_aorta_diameter_mm_{sex}"] = spearman([r["age"] for r in subset],
                                                                          [r.get("aorta_diameter_mm") for r in subset])
    summary["fidelity"]["bmi_vs_liver_volume_ml"] = spearman([r["bmi"] for r in generated],
                                                             [r.get("liver_volume_ml") for r in generated])
    volume = [r["qc"]["labels"]["volume_ml"] for r in ok]
    pieces = [r["qc"]["mesh"]["structures"] for r in ok]
    summary["label_noise"] = {
        "brain_label": sum(bool(v.get("brain")) for v in volume),
        "skull_label": sum(bool(v.get("skull")) for v in volume),
        "reaches_skull_base": sum(v.get("brain", 0) > 20 or v.get("skull", 0) > 50 for v in volume),
        "share_of_meshes_in_pieces_median": round(float(np.median(
            [sum(m["components"] > 1 for m in p.values()) / len(p) for p in pieces])), 3),
        "in_pieces": {name: sum(p.get(name, {}).get("components", 0) > 1 for p in pieces)
                      for name in ("heart", "liver", "aorta", "urinary_bladder", "rib_left_5", "small_bowel")},
    }
    summary["body_cut_sides"] = dict(Counter(side for r in ok for side in r["qc"]["labels"]["body_touching_boundary"]
                                             if side not in ("inferior", "superior")))
    return summary


#: CT reference values by sex (M, F) and source, compared with the generated anatomy in REPORT.md.
REFERENCES: dict[str, tuple[str, float, float, str]] = {
    "liver_volume_ml": ("liver (mL)", 1710, 1411, "Geraghty 2004, mean of 149 CTs"),
    "spleen_volume_ml": ("spleen (mL)", 238.4, 179.8, "Geraghty 2004"),
    "kidney_volume_ml": ("both kidneys (mL)", 325, 269, "Wang 2014, median parenchyma of 1,344 CTs"),
    "lung_volume_ml": ("both lungs (mL)", 6120, 4710, "Bakker 2025, inspiratory CT, 7,306 adults"),
    "aorta_diameter_mm": ("aorta diameter (mm)", 19.3, 16.7, "Rogers 2013, infrarenal (descending thoracic 25.8 / 23.1)"),
}


def reference_table(rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Generated mean ± SD by sex against CT reference values; also returns the ratios."""
    table, ratios = [], {}
    for key, (label, male, female, source) in REFERENCES.items():
        cells = [label]
        for sex, reference in (("M", male), ("F", female)):
            values = [r[key] for r in rows if r["generated"] and r["sex"] == sex and r.get(key)]
            ratios[f"{key}_{sex}"] = round(float(np.mean(values)) / reference, 2) if values else None
            cells += [_mean_sd(values), f"{reference:g}"]
        table.append(cells + [f"{ratios[key + '_M']:.2f} / {ratios[key + '_F']:.2f}", source])
    header = ["measure", "generated men", "reference men", "generated women", "reference women",
              "generated / reference (M / F)", "reference"]
    return markdown(header, table), ratios


def cancel_back_to_back(faces: np.ndarray) -> np.ndarray:
    """Drop triangles whose vertex set occurs more than once: scikit-image's marching cubes emits
    such back-to-back pairs (same vertices, opposite winding), whose edges are non-manifold."""
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.reshape(-1)] == 1]


def mesh_pair_analysis(folder: Path, records: Sequence[Record], ids: Mapping[str, list[int]]) -> dict[str, Any]:
    """Re-mesh every structure flagged as not closed and test whether cancelling back-to-back pairs closes it."""
    from patient_digital_twin.geometry import mask_to_mesh
    from qc import edge_counts
    from scipy import ndimage

    flagged = fixed = with_pairs = 0
    core_records_fixed = 0
    for record in records:
        if record["status"] != "ok" or not record["qc"]["mesh"]["not_closed_manifold"]:
            continue
        labels, _ = canonical(nib.load(str(folder / "patients" / record["id"] / "labels.nii.gz")))
        labels = np.asarray(labels).transpose(2, 1, 0)  # ZYX, as the importers mesh it.
        closed_core = True
        for name in record["qc"]["mesh"]["not_closed_manifold"]:
            mask = np.isin(labels, ids.get(name, []))
            box = ndimage.find_objects(mask.astype(np.uint8))
            if not box or box[0] is None:
                continue
            vertices, faces = mask_to_mesh(mask[box[0]])
            kept = cancel_back_to_back(faces)
            counts = edge_counts(kept, len(vertices))
            flagged += 1
            with_pairs += len(kept) < len(faces)
            ok = bool(len(kept) < len(faces) and (counts == 2).all())
            fixed += ok
            closed_core &= ok or name not in CORE
        core_records_fixed += closed_core and bool(_open_core_meshes(record))
    return {"flagged_meshes": flagged, "with_back_to_back_pairs": with_pairs, "closed_after_cancelling": fixed,
            "patients_whose_core_meshes_close": core_records_fixed}


def characteristic_table(cohort: Record, patients: Sequence[Mapping[str, Any]]) -> str:
    """Characteristic, distribution, sampled range and mean ± SD (overall and by sex), source."""
    from cohort import TRAINING_RANGE

    rows = []
    for entry in cohort["settings"]["spec"]:
        name = entry["characteristic"]
        values = [p[name] for p in patients]
        if isinstance(values[0], str):
            counts = {v: values.count(v) for v in sorted(set(values))}
            sampled = ", ".join(f"{k} {c} ({c / len(values):.1%})" for k, c in counts.items())
            spread, by_sex = "–", "–"
        else:
            sampled = f"{min(values):g}–{max(values):g}"
            spread = _mean_sd(values)
            by_sex = "; ".join(f"{s} {_mean_sd([p[name] for p in patients if p['sex'] == s])}" for s in ("M", "F"))
        training = TRAINING_RANGE.get(name)
        rows.append([f"`{name}`", entry["distribution"], "yes" if entry["conditions_nv_generate"] == "yes" else "no",
                     "–" if training is None else f"{training[0]:g}–{training[1]:g}", sampled, spread, by_sex, entry["source"]])
    return markdown(["characteristic", "distribution (as set)", "conditions NV-Generate", "model training range",
                     "sampled range", "sampled mean ± SD", "by sex", "source"], rows)


def main(argv: Sequence[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--cohort", type=Path, required=True, help="generate_cohort.py output folder")
    cli.add_argument("--sweep", type=Path, help="bmi_sweep.py output folder (optional)")
    cli.add_argument("--out", type=Path, default=HERE, help="Folder for figures/ and results/ (default: this folder)")
    cli.add_argument("--source-root", type=Path, help="NV-Generate-CTMR checkout (label IDs; default from cohort.json)")
    cli.add_argument("--samples", type=int, default=6, help="Patients shown in the sample figures")
    cli.add_argument("--no-render", action="store_true", help="Skip the VTK twin renders")
    args = cli.parse_args(argv)
    from qc import label_ids

    cohort, records = load(args.cohort)
    source_root = args.source_root or Path(cohort["settings"]["source_root"])
    ids = label_ids(source_root)
    refine_completeness(args.cohort, records, ids)
    rows = [flat(r) for r in records]
    spec = PRESETS[cohort["settings"]["preset"]]()
    figures, results = args.out / "figures", args.out / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    summary = summarize(cohort, records, rows)
    summary["settings"] = {k: v for k, v in cohort["settings"].items() if k != "spec"}
    stamps = [p.stat().st_mtime for p in (args.cohort / "patients").glob("*/record.json")]
    started = (args.cohort / "cohort.json").stat().st_mtime
    summary["timing_s"]["wall_hours"] = round((max(stamps) - started) / 3600, 2) if stamps else None
    write_csv(results / "patients.csv", rows)
    comparison, summary["reference_ratios"] = reference_table(rows)
    tables = {  # Title -> Markdown table, for results/tables.md.
        "Characteristics": characteristic_table(cohort, cohort["patients"]),
        "Quality checks": markdown(["criterion", "definition", "passed"],
                                   [[f"`{k}`", v["description"], f"{v['passed']}/{len(rows)}"]
                                    for k, v in {**summary["criteria"], **summary["findings"]}.items()]),
        "BMI classes": markdown(list(summary["bmi_classes"][0]), [list(c.values()) for c in summary["bmi_classes"]]),
        "Generated anatomy against CT references": comparison,
    }
    summary["mesh_pairs"] = mesh_pair_analysis(args.cohort, records, ids)
    figure_distributions(cohort["patients"], spec, figures / "cohort_distributions.png")
    picks = _pick(rows, args.samples)
    summary["samples"] = [p["id"] for p in picks]
    figure_scans(args.cohort, picks, ids, figures / "sample_scans.png")
    if not args.no_render:
        figure_twins(args.cohort, picks, figures / "sample_twins.jpg")
        figure_twin_detail(args.cohort, picks[len(picks) // 2], figures / "twin_detail.jpg")
    figure_fidelity(rows, figures / "conditioning_fidelity.png")
    if args.sweep:
        sweep, sweep_records = load(args.sweep)
        refine_completeness(args.sweep, sweep_records, ids)
        sweep_rows = [flat(r) for r in sweep_records]
        write_csv(results / "bmi_sweep.csv", sweep_rows)
        figure_sweep(args.sweep, sweep_records, ids, figures / "bmi_sweep.png")
        series = {}
        for row in sweep_rows:
            series.setdefault((row["seed"], row["sex"]), []).append(row)
        summary["sweep"] = {
            "settings": sweep, "patients": len(sweep_rows), "generated": sum(r["generated"] for r in sweep_rows),
            "monotonic_adipose": sum(all(np.diff([r["l3_adipose_area_cm2"] for r in sorted(s, key=lambda r: r["bmi"])]) > 0)
                                     for s in series.values() if all(r["generated"] for r in s)),
            "monotonic_body_volume": sum(all(np.diff([r["body_volume_l"] for r in sorted(s, key=lambda r: r["bmi"])]) > 0)
                                         for s in series.values() if all(r["generated"] for r in s)),
            "series": len(series),
            "rho_bmi_adipose": spearman([r["bmi"] for r in sweep_rows if r["generated"]],
                                        [r["l3_adipose_area_cm2"] for r in sweep_rows if r["generated"]]),
            "rho_bmi_body_volume": spearman([r["bmi"] for r in sweep_rows if r["generated"]],
                                            [r["body_volume_l"] for r in sweep_rows if r["generated"]]),
        }
        if not args.no_render:  # The first seed whose male and female series both exported every bundle.
            complete = [seed for seed in sorted({k[0] for k in series}) if all(
                (args.sweep / "patients" / r["id"] / "twin/patient_anatomy.usdc").is_file()
                for sex in ("M", "F") for r in series.get((seed, sex), [None]) if r is not None)
                and all((seed, sex) in series for sex in ("M", "F"))]
            seed = complete[0] if complete else min(k[0] for k in series)
            shown = [r for sex in ("M", "F") for r in sorted(series.get((seed, sex), []), key=lambda r: r["bmi"])]
            summary["sweep"]["rendered_seed"] = seed
            figure_twins(args.sweep, shown, figures / "bmi_sweep_twins.jpg", columns=len(sweep["bmi"]),
                         title=lambda p: f"{p['sex']}, BMI {p['bmi']:g} (seed {p['seed']})")
    (results / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    sections = [f"## {title}\n\n{table}" for title, table in tables.items()]
    (results / "tables.md").write_text("\n\n".join(["# Cohort report tables", *sections]) + "\n")
    print(json.dumps({k: summary[k] for k in ("patients", "generated", "criteria")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
