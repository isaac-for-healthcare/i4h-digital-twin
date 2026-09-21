# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared example-only viewer setup and honest import/containment validation."""

import argparse
import json
import threading
from pathlib import Path

import numpy as np
from viewer import HumanBodyViewer


def parser(description):
    """Common example controls; explicit registration can handle partial scans."""
    result = argparse.ArgumentParser(description=description)
    result.add_argument(
        "--parameters",
        type=Path,
        help="JSON attach_soma arguments for a partial scan or an existing fit",
    )
    result.add_argument("--port", type=int, default=8080)
    result.add_argument("--validate-only", action="store_true")
    result.add_argument(
        "--partial-preview",
        action="store_true",
        help="For a scan with two hip landmarks only, use their width and assume RAS +Z is superior; preview registration only",
    )
    result.add_argument(
        "--fit-shape",
        action="store_true",
        help="Fit SOMA skin around the rigid imported anatomy at the scan pose",
    )
    result.add_argument("--report", type=Path, default=Path("importer_report.json"))
    return result


def show(importer, args):
    """Return a report proving scene completeness, with separate containment results."""
    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    body = importer.to_human_body()
    return show_body(body, importer.report, args)


def partial_registration(body):
    """Preview a hip-only RAS scan; use explicit parameters for measured alignment."""
    left, right = (np.asarray(body.landmarks[key]) for key in ("left_hip", "right_hip"))

    def frame(a, b, up):
        x = a - b
        if np.linalg.norm(x) < 1e-6:
            raise ValueError("Degenerate hip landmarks")
        x /= np.linalg.norm(x)
        y = np.asarray(up, dtype=float) - x * np.dot(up, x)
        if np.linalg.norm(y) < 1e-6:
            raise ValueError("Hip axis cannot be parallel to the assumed superior axis")
        y /= np.linalg.norm(y)
        return np.column_stack([x, y, np.cross(x, y)])

    source = frame(left, right, [0, 0, 1])
    body.attach_soma(body_to_soma=np.eye(4), refine_limbs=False)
    target_left, target_right = (
        body.soma_body.joints[j] for j in ("LeftLeg", "RightLeg")
    )
    target = frame(target_left, target_right, [0, 1, 0])
    scale = np.linalg.norm(left - right) / np.linalg.norm(target_left - target_right)
    matrix = np.eye(4)
    matrix[:3, :3] = target @ source.T
    matrix[:3, 3] = scale * (target_left + target_right) / 2 - matrix[:3, :3] @ (
        (left + right) / 2
    )
    body.attach_soma(
        body.soma_layer,
        body_to_soma=matrix,
        global_scale=float(scale),
        refine_limbs=False,
    )


def show_body(body, import_report, args):
    """Open an imported body; no missing mesh or containment failure is hidden."""
    report = {"import": import_report}
    params = (
        json.loads(args.parameters.read_text())
        if args.parameters
        else {"refine_limbs": False}
    )
    try:
        trunk = {
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        } & body.landmarks.keys()
        if (
            not args.parameters
            and args.partial_preview
            and len(trunk) < 3
            and {"left_hip", "right_hip"} <= trunk
        ):
            partial_registration(body)
            report["alignment_method"] = (
                "Preview: paired hips with assumed RAS superior direction"
            )
        else:
            body.attach_soma(**params)
            report["alignment_method"] = (
                "Explicit parameters" if args.parameters else "Bone-landmark matching"
            )
    except ValueError as exc:
        report["alignment_error"] = str(exc)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        raise ValueError(
            f"{exc}. Supply --parameters with measured body-frame landmarks or body_to_soma. Import coverage saved to {args.report}."
        ) from exc
    if args.fit_shape:
        report["shape_fit"] = body.fit_soma_shape(components=12, iterations=12)
    report["alignment_errors_m"] = body.alignment_errors_m
    viewer = HumanBodyViewer(body, port=0 if args.validate_only else args.port)
    try:
        expected = {
            name
            for name, structure in body.structures.items()
            if not structure.is_empty
        }
        visible = {name for name, mesh in viewer.meshes.items() if mesh.visible}
        report["viewer"] = {
            "expected": sorted(expected),
            "visible": sorted(visible),
            "all_expected_visible": visible == expected,
        }
        report["containment"] = body.check_containment()
        report["all_inside"] = bool(expected) and all(
            item["passed"] for item in report["containment"].values()
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{len(visible)}/{len(expected)} imported anatomy meshes visible; all inside skin: {report['all_inside']}. Report: {args.report}",
            flush=True,
        )
        if args.validate_only:
            return report
        print(f"Open http://127.0.0.1:{viewer.server.get_port()}", flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    finally:
        viewer.close()
    return report
