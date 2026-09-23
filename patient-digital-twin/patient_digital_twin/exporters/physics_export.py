# SPDX-License-Identifier: Apache-2.0
"""Export existing patient bundles for the unmodified OmniEndo/OmniSurg demos.

The physics checkout supplies configuration and instruments only. Every anatomy
vertex comes from the input bundle. Scene-unit transforms are recorded explicitly.
"""

import os
import shutil
import time
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml


def _mesh(stage, prim_path):
    import trimesh
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    mesh = UsdGeom.Mesh(prim)
    if not mesh or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
        raise ValueError(f"Missing or disabled patient mesh: {prim_path}")
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    points = (
        points @ matrix[:3, :3].T + matrix[:3, 3]
    ) * UsdGeom.GetStageMetersPerUnit(stage)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    faces, offset = [], 0
    for count in counts:
        faces.extend(
            (indices[offset], indices[offset + i], indices[offset + i + 1])
            for i in range(1, count - 1)
        )
        offset += count
    result = trimesh.Trimesh(points, faces, process=True)
    result.fix_normals()
    return max(result.split(only_watertight=False), key=lambda m: len(m.faces))


def _write_yaml(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _write_usd(path, mesh):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    prim = UsdGeom.Mesh.Define(stage, "/Anatomy")
    stage.SetDefaultPrim(prim.GetPrim())
    prim.CreatePointsAttr(mesh.vertices.tolist())
    prim.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    prim.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    prim.CreateSubdivisionSchemeAttr("none")
    stage.GetRootLayer().Save()


def _xcath(mesh, template, folder, name):
    import trimesh

    # Preserve anatomy proportions while using the existing demo's numerical
    # scene scale. This is not a claim that the authored demo uses SI parameters.
    scale = 30.0
    sign = 1 if name == "aorta" else -1
    z = mesh.bounds[0, 2] + 0.003 if sign == 1 else mesh.bounds[1, 2] - 0.003
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        raise ValueError(f"No entry cross-section in {name}")
    loop = max(section.discrete, key=lambda p: len(p))
    entry = np.r_[loop[:, :2].mean(axis=0), z]
    cropped = trimesh.intersections.slice_mesh_plane(
        mesh, [0, 0, sign], entry, cap=False
    )
    cropped.merge_vertices()
    cropped.remove_unreferenced_vertices()
    rotation = np.array([[0, 0, sign], [1, 0, 0], [0, sign, 0]], dtype=float)
    scene = deepcopy(template)
    catheter = scene["catheters"][0]
    length = (catheter["centerline"]["point_count"] - 1) * catheter["centerline"][
        "segment_length"
    ]
    transform = np.eye(4)
    transform[:3, :3] = rotation * scale
    transform[:3, 3] = np.array([length - 0.1, 0, 1]) - transform[:3, :3] @ entry
    cropped.apply_transform(transform)
    cropped.vertices = np.asarray(cropped.vertices, dtype=np.float32)
    cropped.merge_vertices(digits_vertex=5)
    cropped.update_faces(cropped.unique_faces())
    cropped.update_faces(cropped.nondegenerate_faces(height=1e-5))
    cropped.remove_unreferenced_vertices()
    _write_usd(folder / f"{name}.usda", cropped)
    vessel = scene["vessels"][0]
    vessel["source"] = {"path": f"{name}.usda", "prim_path": "/Anatomy"}
    vessel["transform"] = {
        "scale": 1.0,
        "translation": [0.0, 0.0, 0.0],
        "rotation_euler_degrees": [0.0, 0.0, 0.0],
    }
    vessel.pop("centerline", None)  # Never reuse the stock anatomy's centerline.
    vessel["display"]["hide_external_faces"] = False
    scene["camera"] = {"mode": "auto_frame", "fov": 45.0}
    _write_yaml(folder / f"{name}.yaml", scene)
    return {
        "config": f"physics/{name}.yaml",
        "source_structure": "aorta" if name == "aorta" else "trachea",
        "scene_from_patient_m": transform.tolist(),
        "entry_patient_m": entry.tolist(),
        "open_entry_cut_m": 0.003,
        "vertices": len(cropped.vertices),
        "triangles": len(cropped.faces),
    }


def _liver(mesh, root, folder):
    try:
        import tetgen
    except ImportError as exc:
        raise ImportError(
            "Liver physics export requires patient-digital-twin[physics]"
        ) from exc
    transform = np.eye(4)
    transform[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) * 20.0
    transform[:3, 3] = np.array([0, 2, -5.5]) - transform[:3, :3] @ mesh.bounds.mean(
        axis=0
    )
    mesh.apply_transform(transform)
    generator = tetgen.TetGen(
        np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)
    )
    nodes, tets, *_ = generator.tetrahedralize(
        order=1, quality=False, steinerleft=20000
    )
    # Explicitly orient positive-volume elements and derive the outward boundary.
    determinants = np.linalg.det(nodes[tets[:, 1:]] - nodes[tets[:, 0, None]])
    negative = determinants < 0
    tets[negative] = tets[negative][:, [1, 0, 2, 3]]
    all_faces = tets[:, np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])].reshape(
        -1, 3
    )
    _, index, counts = np.unique(
        np.sort(all_faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    faces = all_faces[index[counts == 1]]
    edges = np.unique(
        np.sort(
            tets[:, np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])].reshape(
                -1, 2
            ),
            axis=1,
        ),
        axis=0,
    )
    bundle = folder / "liver_tet"
    bundle.mkdir()
    for name, values, fmt in [
        ("vertices", nodes, "%.9g"),
        ("tetras", tets, "%d"),
        ("edges", edges, "%d"),
        ("tris", faces, "%d"),
    ]:
        np.savetxt(bundle / f"model.{name}", values, fmt=fmt)
    examples = root / "physics_simulation/surgical/examples"
    scene = yaml.safe_load((examples / "liver.yaml").read_text())
    assets = yaml.safe_load((examples / "liver/assets.yaml").read_text())
    asset = assets["assets"][0]
    asset["source"] = "liver_tet"
    settings = asset["parameters"]
    settings.update(
        data_path="liver_tet",
        translation=[0.0, 0.0, 0.0],
        bounds_min=(nodes.min(0) - 2).tolist(),
        bounds_max=(nodes.max(0) + 2).tolist(),
        pin_center=nodes[nodes[:, 1].argmax()].tolist(),
        pin_radius=0.5,
    )
    _write_yaml(folder / "liver_assets.yaml", assets)
    scene["includes"]["assets"] = ["liver_assets.yaml"]
    instruments = yaml.safe_load((examples / "liver/instruments.yaml").read_text())
    for instrument in instruments["instruments"]:
        instrument["parameters"]["mesh_scale"] = 0.0075
    _write_yaml(folder / "liver_instruments.yaml", instruments)
    scene["includes"]["instruments"] = ["liver_instruments.yaml"]
    for key in ["mechanical_materials"]:
        scene["includes"][key] = [
            str((examples / p).resolve()) for p in scene["includes"][key]
        ]
    scene["includes"]["visual_materials"] = [
        str(examples / "liver/visual_materials.yaml")
    ]
    solver = yaml.safe_load((examples / "liver/solver.yaml").read_text())
    solver["systems"].insert(
        1,
        {
            "id": "patient_liver_attachments",
            "type": "tet.attachments",
            "body": "liver_tet",
            "enable": True,
            "requires_after": ["liver_tet_deformation"],
        },
    )
    _write_yaml(folder / "liver_solver.yaml", solver)
    scene["includes"]["solver"] = "liver_solver.yaml"
    for device in scene["inputs"]["devices"]:
        device["backend"] = "fallback"  # No haptic hardware is assumed for the demo.
    render = scene["rendering"]["settings"]
    render.pop("environment_path", None)
    render.pop("postprocess_path", None)
    render["width"] = 2200
    render["height"] = 1100
    render["camera"] = {
        "position": [0.0, 5.0, 2.0],
        "pitch": -20.0,
        "yaw": -90.0,
        "fov": 45.0,
    }
    _write_yaml(folder / "liver.yaml", scene)
    return {
        "config": "physics/liver.yaml",
        "source_structure": "liver",
        "scene_from_patient_m": transform.tolist(),
        "vertices": len(nodes),
        "tetrahedra": len(tets),
        "triangles": len(faces),
        "minimum_tet_volume": float(np.abs(determinants).min() / 6),
    }


def export_physics_examples(patient_twin, output, *, physics_root, demos=None):
    """Write an extended patient manifest and demo inputs in a new directory.

    Existing artifacts are referenced, not copied or changed. The physics source
    checkout is read-only. Optional dependencies: usd-core, trimesh,
    fast-simplification and tetgen. Airways use the retained trachea segmentation.
    """
    from pxr import Usd

    source = Path(patient_twin).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    root = Path(physics_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    manifest = yaml.safe_load(source.read_text())
    anatomy = manifest["anatomy"]["structures"]
    sources = {"aorta": "aorta", "airways": "trachea", "liver": "liver"}
    demos = tuple(sources) if demos is None else tuple(demos)
    if not demos or set(demos) - sources.keys():
        raise ValueError("Select one or more physics demos: aorta, airways, liver")
    for name in (sources[demo] for demo in demos):
        if name not in anatomy or not anatomy[name].get("enabled", True):
            raise ValueError(f"Missing or disabled patient anatomy: {name}")
    stage = Usd.Stage.Open(str(source.parent / manifest["artifacts"]["anatomy_usd"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        folder = Path(temporary)
        physics = folder / "physics"
        physics.mkdir()
        results = {}
        for demo in demos:
            started = time.perf_counter()
            name = sources[demo]
            mesh = _mesh(stage, anatomy[name]["prim_path"])
            original_faces = len(mesh.faces)
            if len(mesh.faces) > 6000:
                mesh = mesh.simplify_quadric_decimation(face_count=6000)
                mesh.fix_normals()
            import pymeshfix
            import trimesh

            vertices, faces = pymeshfix.clean_from_arrays(
                np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                np.ascontiguousarray(mesh.faces, dtype=np.int32),
            )
            mesh = trimesh.Trimesh(vertices, faces, process=True)
            mesh.fix_normals()
            if demo == "liver":
                results[demo] = _liver(mesh, root, physics)
            else:
                template = yaml.safe_load(
                    (
                        root
                        / f"physics_simulation/endoluminal/xcath/scenes/{demo}.yaml"
                    ).read_text()
                )
                results[demo] = _xcath(mesh, template, physics, demo)
            results[demo]["source_triangles"] = original_faces
            results[demo]["generation_seconds"] = time.perf_counter() - started
        manifest["physics_examples"] = {
            "source_patient_twin": os.path.relpath(source, output),
            "demos": results,
        }
        for key, value in manifest["artifacts"].items():
            manifest["artifacts"][key] = os.path.relpath(
                (source.parent / value).resolve(), output
            )
        _write_yaml(folder / "patient_twin.yaml", manifest)
        shutil.move(str(folder), str(output))
    return output / "patient_twin.yaml"
