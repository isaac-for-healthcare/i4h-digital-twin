# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent CT-to-centerline pipelines: TotalSegmentator versus NV-Segment.

Neither pipeline reads the subject's supplied segmentation files. Both compare
an aorta/iliac-artery tree in canonical LPS coordinates, using the voxel thinning
and closing settings used by i4h-workflows. Inference outputs are never shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from patient_digital_twin import AnatomicalStructure, CenterlineGraph, Kind
from patient_digital_twin.imaging_to_mesh import mask_to_mesh
from patient_digital_twin.importers import NVSegmentImporter
from patient_digital_twin.topology import voxelize_mesh
from scipy import ndimage
from scipy.spatial import cKDTree

LABELS = ("aorta", "iliac_artery_left", "iliac_artery_right")


def _prepare(mask):
    """Apply the same fixed workflow postprocessing independently to each result."""
    from i4h_tools.patient_twin import pipeline

    closed = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)), iterations=2)
    return pipeline._largest_component(closed)


def _ct(path):
    from i4h_tools.patient_twin import pipeline

    return pipeline._ingest(Path(path))


def _grid(ct):
    return {
        "shape_zyx": ct.hu_zyx.shape,
        "spacing_zyx_m": np.asarray(ct.spacing_zyx_mm) * 0.001,
        "origin_xyz_m": np.asarray(ct.origin_xyz_mm) * 0.001,
    }


def vasculature_digital_twin_pipeline(ct_path):
    """CT → real TotalSegmentator inference → workflow vasculature centerline."""
    import nibabel as nib
    from i4h_tools.patient_twin import pipeline
    from totalsegmentator.map_to_binary import class_map
    from vasculature_digital_twin.vasculature import vessel_mask_from_totalsegmentator

    ct = _ct(ct_path)
    # No cache and no HU-threshold fallback: segmentation must actually run.
    result = vessel_mask_from_totalsegmentator(
        nifti_input=str(ct_path), device="gpu", fast=False
    )
    source = nib.load(str(ct_path))
    # The backend inverts preprocessing to the original NIfTI grid. Reorient the
    # integer labels to the same canonical LPS grid as the ingested CT.
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(source.affine),
        nib.orientations.axcodes2ornt(("L", "P", "S")),
    )
    labels_xyz = nib.orientations.apply_orientation(
        result.label_map.transpose(2, 1, 0), orientation
    )
    labelmap = {name: i for i, name in class_map["total"].items()}
    mask = np.isin(labels_xyz.transpose(2, 1, 0), [labelmap[n] for n in LABELS])
    if mask.shape != ct.hu_zyx.shape:
        raise ValueError("TotalSegmentator output does not match the CT grid")
    present = [n for n in LABELS if np.any(result.label_map == labelmap[n])]
    if tuple(present) != LABELS:
        raise ValueError(
            f"TotalSegmentator missing comparison vessels: {set(LABELS) - set(present)}"
        )
    mask = _prepare(mask)
    graph = pipeline._centerline(mask, ct.spacing_zyx_mm, ct.origin_xyz_mm)
    return (
        graph,
        mask,
        {"backend": "TotalSegmentator", "labels": present, "inference_ran": True},
    )


def patient_digital_twin_pipeline(ct_path, *, bundle_root=None, python_executable=None):
    """CT → NVSegmentImporter → HumanBody meshes → HumanBody centerline."""
    ct = _ct(ct_path)
    importer = NVSegmentImporter(
        ct_path, bundle_root=bundle_root, python_executable=python_executable
    )
    body = importer.to_human_body()
    grid = _grid(ct)
    mask = np.zeros(ct.hu_zyx.shape, dtype=bool)
    for name in LABELS:
        structure = body.structures.get(name)
        if structure is None or structure.is_empty:
            raise ValueError(f"NV-Segment missing comparison vessel: {name}")
        # Importer retains RAS imaging coordinates in meters. Voxelize its own
        # inferred mesh in canonical LPS without consulting the other pipeline.
        vertices_lps = body.imaging_vertices(name) * [-1, -1, 1]
        mask |= voxelize_mesh(vertices_lps, structure.faces, **grid)
    mask = _prepare(mask)
    vertices, faces = mask_to_mesh(
        mask, spacing_zyx_mm=ct.spacing_zyx_mm, origin_xyz_mm=ct.origin_xyz_mm
    )
    local_to_body = np.linalg.inv(body.body_to_imaging) @ np.diag(
        [-1.0, -1.0, 1.0, 1.0]
    )
    body.structures["vascular_tree"] = AnatomicalStructure(
        "vascular_tree",
        Kind.VESSEL,
        vertices.astype(float) * 0.001,
        faces,
        local_to_body=local_to_body,
    )
    body.extract_topology(names=["vascular_tree"], method="skeleton", **grid)
    graph = body.structures["vascular_tree"].centerline
    # Its local frame is explicitly LPS meters; posing remains a rigid transform.
    return (
        graph,
        mask,
        {
            "backend": "NV-Segment",
            "labels": list(LABELS),
            "inference_ran": True,
            "import_coverage": importer.report,
        },
    )


def _samples(graph, scale, step_mm=0.5):
    """Uniform edge sampling avoids a comparison dominated by node density."""
    segments = np.asarray(graph.points)[graph.edges] * scale
    samples = []
    for a, b in segments:
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / step_mm)))
        samples.append(a + np.arange(count)[:, None] / count * (b - a))
    if not samples:
        raise ValueError("Cannot compare a centerline without edges")
    return np.concatenate(samples)


def compare_graphs(reference, actual, *, tolerance_mm=3.0, required_coverage=0.95):
    """Symmetric distance/coverage; independent segmenters need not share nodes."""
    ref = _samples(reference, 1.0)
    new = _samples(actual, 1000.0)
    forward = cKDTree(new).query(ref)[0]
    reverse = cKDTree(ref).query(new)[0]
    distances = np.concatenate([forward, reverse])
    coverage_ref = float(np.mean(forward <= tolerance_mm))
    coverage_new = float(np.mean(reverse <= tolerance_mm))
    return {
        "reference_points": len(reference.points),
        "human_body_points": len(actual.points),
        "reference_edges": len(reference.edges),
        "human_body_edges": len(actual.edges),
        "mean_symmetric_distance_mm": float(np.mean(distances)),
        "p95_symmetric_distance_mm": float(np.percentile(distances, 95)),
        "max_symmetric_distance_mm": float(np.max(distances)),
        "reference_coverage": coverage_ref,
        "human_body_coverage": coverage_new,
        "tolerance_mm": tolerance_mm,
        "required_coverage": required_coverage,
        "sampling_step_mm": 0.5,
        "similar": min(coverage_ref, coverage_new) >= required_coverage,
    }


def _save(output, prefix, graph, mask, metadata):
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"{prefix}.npz",
        points=graph.points,
        radii=graph.radii,
        edges=graph.edges,
        mask=mask,
    )
    (output / f"{prefix}.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _load(output, prefix):
    with np.load(output / f"{prefix}.npz") as data:
        return CenterlineGraph(data["points"], data["radii"], data["edges"]), data[
            "mask"
        ]


def run(
    ct_path,
    output,
    *,
    bundle_root=None,
    python_executable=None,
    stage="both",
    tolerance_mm=3.0,
    required_coverage=0.95,
):
    ct_path, output = ct_path.expanduser().resolve(), output.expanduser().resolve()
    scan_hash = hashlib.sha256(ct_path.read_bytes()).hexdigest()
    for name in ["vasculature", "patient"]:
        if stage not in ("both", name):
            continue
        print(f"Running independent {name} CT pipeline", flush=True)
        if name == "vasculature":
            graph, mask, metadata = vasculature_digital_twin_pipeline(ct_path)
        else:
            graph, mask, metadata = patient_digital_twin_pipeline(
                ct_path, bundle_root=bundle_root, python_executable=python_executable
            )
        metadata.update(
            ct=str(ct_path),
            ct_sha256=scan_hash,
            coordinate_frame="LPS",
            graph_units="mm" if name == "vasculature" else "m",
        )
        _save(output, name, graph, mask, metadata)
    if stage not in ("both", "compare"):
        return
    metadata = {
        n: json.loads((output / f"{n}.json").read_text())
        for n in ["vasculature", "patient"]
    }
    if any(m["ct_sha256"] != scan_hash for m in metadata.values()):
        raise ValueError("Pipeline artifacts do not originate from this CT")
    reference, ref_mask = _load(output, "vasculature")
    actual, new_mask = _load(output, "patient")
    report = compare_graphs(
        reference,
        actual,
        tolerance_mm=tolerance_mm,
        required_coverage=required_coverage,
    )
    report.update(
        segmentation_dice=float(
            2
            * np.count_nonzero(ref_mask & new_mask)
            / (ref_mask.sum() + new_mask.sum())
        ),
        pipelines=metadata,
        labels=list(LABELS),
        ct=str(ct_path),
        postprocessing="Independent union, closing (2), largest component, voxel skeletonization",
    )
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_comparison(_ct(ct_path), reference, actual, output / "side_by_side.png")
    print(json.dumps(report, indent=2), flush=True)
    if not report["similar"]:
        raise RuntimeError(
            "Independent predictions did not meet the preselected similarity threshold; see comparison.json"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct",
        type=Path,
        default=Path("~/dev/data/Totalsegmentator_dataset_small_v201/s0011/ct.nii.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data/i4h_workflows_s0011_independent",
    )
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--python", dest="python_executable")
    parser.add_argument(
        "--stage", choices=["both", "vasculature", "patient", "compare"], default="both"
    )
    parser.add_argument("--tolerance-mm", type=float, default=3.0)
    parser.add_argument("--required-coverage", type=float, default=0.95)
    args = parser.parse_args()
    run(
        args.ct,
        args.output,
        bundle_root=args.bundle_root,
        python_executable=args.python_executable,
        stage=args.stage,
        tolerance_mm=args.tolerance_mm,
        required_coverage=args.required_coverage,
    )


def plot_comparison(ct, reference, actual, output):
    """Save side-by-side centerlines over the same coronal CT maximum projection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    z, _, x = ct.hu_zyx.shape
    sz, _, sx = ct.spacing_zyx_mm
    ox, _, oz = ct.origin_xyz_mm
    extent = [ox - sx / 2, ox + (x - 0.5) * sx, oz - sz / 2, oz + (z - 0.5) * sz]
    projection = np.max(ct.hu_zyx, axis=1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, points, edges, color, title in [
        (
            axes[0],
            reference.points,
            reference.edges,
            "tomato",
            "TotalSegmentator → vasculature_digital_twin",
        ),
        (
            axes[1],
            actual.points * 1000,
            actual.edges,
            "deepskyblue",
            "NV-Segment → HumanBody → centerline",
        ),
    ]:
        ax.imshow(
            projection, origin="lower", extent=extent, cmap="gray", vmin=-200, vmax=1000
        )
        ax.add_collection(
            LineCollection(points[edges][:, :, [0, 2]], colors=color, linewidths=1.5)
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("LPS X (mm)")
        ax.set_ylabel("LPS Z (mm)")
    figure.suptitle("s0011 · aorta + iliac arteries · coronal CT maximum projection")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
