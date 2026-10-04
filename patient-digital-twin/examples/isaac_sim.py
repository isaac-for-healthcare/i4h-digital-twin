# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export s0011 anatomy to USD, then view it with an installed Isaac Sim runtime.

``python examples/isaac_sim.py export out.usdc --anatomy aorta liver --with-ct`` meshes the
bundled s0011 masks (``generate_body``); ``<isaac-sim>/python.sh examples/isaac_sim.py view
out.usdc`` opens the stage in Isaac Sim with a session-only light and camera (``view_usd``),
optionally headless for a fixed number of frames with a PNG screenshot.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

SAMPLE = Path(__file__).resolve().parent / "data/s0011"


def generate_body(output: str | Path, *, names: Sequence[str] = ("aorta",), with_ct: bool = False) -> Path:
    """Mesh the named s0011 masks, optionally embed the matching CT, and write a USD; returns its path."""
    from patient_digital_twin import HumanBody, SegmentationImporter
    from patient_digital_twin.scan_volume import from_nifti

    body = HumanBody(SegmentationImporter(SAMPLE / "segmentations", names=names).to_anatomy_collection())
    if with_ct:
        body.attach_scan(from_nifti(SAMPLE / "ct.nii.gz"))
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return body.export_to_usd(output)


def view_usd(path: str | Path, *, headless: bool = False, frames: int = 0, screenshot: str | Path | None = None) -> None:
    """Open a patient USD in Isaac Sim and frame its anatomy; lighting/camera edits stay session-only.

    ``frames=0`` runs until the window closes (not allowed headless); ``screenshot`` saves a PNG.
    """
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


def main() -> None:
    """CLI entry point for the ``export`` and ``view`` actions."""
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
