# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a HumanBody USD, then open it using Isaac Sim's Python runtime.

python examples/isaac_sim.py export /tmp/human_body.usdc
/path/to/isaac-sim/python.sh examples/isaac_sim.py view /tmp/human_body.usdc
"""

import argparse
from pathlib import Path

SAMPLE = Path(__file__).resolve().parent / "data/nv_ct_high_resolution"


def generate_body(output):
    """Generate meshes from the sample segmentation and export USD."""
    from patient_digital_twin import HumanBody, SegmentationImporter

    importer = SegmentationImporter(
        SAMPLE / "segmentation.nii.gz", SAMPLE / "labels.json"
    )
    body = HumanBody(importer.to_anatomy_collection())
    body.anatomy.configure(SAMPLE / "anatomy.yaml")
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return body.export_to_usd(output)


def view_usd(path):
    """Open the exported static stage and render until the viewer is closed."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": False})
    try:
        # Kit must be running before importing its USD and viewport modules.
        import omni.usd
        from isaacsim.core.utils.stage import is_stage_loading, open_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Usd, UsdGeom, UsdLux

        if not open_stage(str(path)):
            raise RuntimeError(f"Could not open USD stage: {path}")
        app.update()
        while app.is_running() and is_stage_loading():
            app.update()
        if not app.is_running():
            return
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath("/HumanBody")
        if not root.IsValid():
            raise ValueError("Expected a HumanBody export with /HumanBody")
        # Viewer-only lighting and camera edits are not saved to the asset.
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdLux.DomeLight.Define(stage, "/ViewerLight").CreateIntensityAttr(1000)
        bounds = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            .ComputeWorldBound(root)
            .ComputeAlignedRange()
        )
        center = bounds.GetMidpoint()
        distance = max(bounds.GetSize().GetLength(), 1.0)
        set_camera_view(
            eye=[center[0] + distance, center[1] - distance, center[2] + distance],
            target=list(center),
        )
        while app.is_running():
            app.update()
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "view"))
    parser.add_argument("usd", type=Path, help="Output/input USD path")
    args = parser.parse_args()
    if args.action == "export":
        print(generate_body(args.usd))
    else:
        view_usd(args.usd)


if __name__ == "__main__":
    main()
