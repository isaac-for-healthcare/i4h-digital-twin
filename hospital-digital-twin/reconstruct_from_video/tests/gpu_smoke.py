# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run inside the 3dgrut image, with /workspace as the working directory.

Exercises CUDA kernels, training, and USDZ export on a small synthetic COLMAP
fixture. This verifies execution, not reconstruction quality on real imagery.
"""

import math
import random
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from pxr import Usd


def main():
    assert torch.__version__.split("+")[0] == "2.7.1", torch.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(), flush=True)
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    a = torch.randn(128, 128, device="cuda", requires_grad=True)
    loss = (a @ a.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()

    with tempfile.TemporaryDirectory(prefix="3dgrut-smoke-") as tmp:
        root = Path(tmp)
        images = root / "colmap/images"
        sparse = root / "colmap/sparse/0"
        images.mkdir(parents=True)
        sparse.mkdir(parents=True)
        rng = random.Random(42)
        points = [
            (rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(2.5, 3.5),
             rng.randrange(50, 256), rng.randrange(50, 256), rng.randrange(50, 256))
            for _ in range(512)
        ]
        (sparse / "cameras.txt").write_text("1 PINHOLE 128 128 100 100 64 64\n")
        (sparse / "points3D.txt").write_text("".join(
            f"{i} {x} {y} {z} {r} {g} {b} 0\n"
            for i, (x, y, z, r, g, b) in enumerate(points, 1)
        ))
        poses = []
        for i in range(16):
            angle = 2 * math.pi * i / 16
            cx, cy = 0.35 * math.cos(angle), 0.35 * math.sin(angle)
            name = f"{i:03d}.png"
            image = Image.new("RGB", (128, 128))
            draw = ImageDraw.Draw(image)
            for x, y, z, r, g, b in sorted(points, key=lambda p: -p[2]):
                u, v = 100 * (x - cx) / z + 64, 100 * (y - cy) / z + 64
                draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(r, g, b))
            image.save(images / name)
            poses.append(f"{i + 1} 1 0 0 0 {-cx} {-cy} 0 1 {name}\n\n")
        (sparse / "images.txt").write_text("".join(poses))
        subprocess.run([
            sys.executable, "-u", "train.py", "--config-name", "apps/colmap_3dgut_mcmc.yaml",
            f"path={root / 'colmap'}", f"out_dir={root / 'out'}", "experiment_name=gpu_smoke",
            "n_iterations=10", "num_workers=0", "enable_writer=false", "test_last=false",
            "compute_extra_metrics=false", "export_ingp.enabled=false",
            "export_usdz.enabled=true", "export_usdz.apply_normalizing_transform=true",
        ], check=True)
        exports = list((root / "out").rglob("export_last.usdz"))
        assert len(exports) == 1, exports
        with zipfile.ZipFile(exports[0]) as archive:
            assert archive.testzip() is None
        stage = Usd.Stage.Open(str(exports[0]))
        assert stage and any(stage.Traverse()), "USDZ contains no readable prims"
        print("PASS: CUDA forward/backward, 10 training steps, readable USDZ export", flush=True)


if __name__ == "__main__":
    main()
