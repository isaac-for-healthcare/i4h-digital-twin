# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export s0011 anatomy, then view it with an installed Isaac Sim runtime."""

import argparse
import math
from pathlib import Path

SAMPLE = Path(__file__).resolve().parent / "data/s0011"


def generate_body(output, *, names=("aorta",), with_ct=False):
    """Extract selected supplied s0011 masks and optionally embed the matching CT."""
    from patient_digital_twin import HumanBody, SegmentationImporter
    from patient_digital_twin.scan_volume import from_nifti

    body = HumanBody(SegmentationImporter(SAMPLE / "segmentations", names=names).to_anatomy_collection())
    if with_ct:
        body.AttachScan(from_nifti(SAMPLE / "ct.nii.gz"))
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return body.export_to_usd(output)


def view_usd(path, *, headless=False, frames=0, screenshot=None):
    """Frame source-unit geometry; keep all lighting/camera edits session-only."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if frames < 0 or (headless and frames == 0):
        raise ValueError("Headless viewing requires a positive --frames limit")
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": headless, "width": 960, "height": 720})
    try:
        import omni.usd
        from isaacsim.core.utils.stage import is_stage_loading, open_stage
        from pxr import Gf, Usd, UsdGeom, UsdLux

        if not open_stage(str(path)):
            raise RuntimeError(f"Could not open USD stage: {path}")
        for _ in range(2000):
            app.update()
            if not is_stage_loading():
                break
        else:
            raise TimeoutError("USD stage did not finish loading")
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/HumanBody")
        if not root.IsValid():
            raise ValueError("Expected a HumanBody export with /HumanBody")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(root).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise ValueError("Patient stage has no visible geometry")
        center = bounds.GetMidpoint()
        unit = UsdGeom.GetStageMetersPerUnit(stage)
        radius = max(bounds.GetSize().GetLength() / 2, 0.005 / unit)
        # Camera coordinates and clip planes use stage units, which can be mm or m.
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdLux.DomeLight.Define(stage, "/ViewerLight").CreateIntensityAttr(500)
        camera = UsdGeom.Camera.Define(stage, "/ViewerCamera")
        half_fov = math.atan(min(camera.GetHorizontalApertureAttr().Get(), camera.GetVerticalApertureAttr().Get()) / (2 * camera.GetFocalLengthAttr().Get()))
        distance = 1.2 * radius / math.sin(half_fov)
        offset = distance / math.sqrt(3)
        eye = center + Gf.Vec3d(offset, -offset, offset)
        camera.MakeMatrixXform().Set(Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0, 0, 1)).GetInverse())
        camera.CreateClippingRangeAttr(Gf.Vec2f(distance / 10000, distance * 10))
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera.GetPath()
        if screenshot is not None:
            import numpy as np
            import omni.replicator.core as rep
            from PIL import Image

            product = rep.create.render_product(str(camera.GetPath()), (960, 720))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach(product)
            rep.orchestrator.step(rt_subframes=8)
            pixels = rgb.get_data()
            if pixels.size == 0 or not np.isfinite(pixels).all() or pixels[..., :3].std() == 0:
                raise RuntimeError("Viewer produced an empty or uniform image")
            screenshot = Path(screenshot)
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(screenshot)
            rgb.detach(product)
            product.destroy()
        rendered = 0
        while app.is_running() and (not frames or rendered < frames):
            app.update()
            rendered += 1
        print(f"Viewed {path}: {rendered} frames, metersPerUnit={unit}, bounds={bounds}", flush=True)
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "view"))
    parser.add_argument("usd", type=Path, help="Output/input USD path")
    parser.add_argument("--anatomy", nargs="+", default=["aorta"], help="s0011 masks to export")
    parser.add_argument("--with-ct", action="store_true", help="Embed native HU CT during export")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Viewer frame limit; 0 runs until closed")
    parser.add_argument("--screenshot", type=Path, help="Save a rendered PNG (requires Pillow)")
    args = parser.parse_args()
    if args.action == "export":
        print(generate_body(args.usd, names=args.anatomy, with_ct=args.with_ct))
    else:
        view_usd(args.usd, headless=args.headless, frames=args.frames, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
