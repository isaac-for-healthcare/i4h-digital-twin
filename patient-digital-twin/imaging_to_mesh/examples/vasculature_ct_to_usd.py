# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a synthetic contrast CT with vasculature_digital_twin and export USD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from imaging_to_mesh import convert_mask_to_usd, convert_segmentation_array
from vasculature_digital_twin import VolumePreprocessor, get_vessel_mask


def synthetic_contrast_ct(shape: tuple[int, int, int] = (72, 72, 72)) -> np.ndarray:
    """Create a small branching, contrast-enhanced CT volume in HU."""
    hu = np.full(shape, -900.0, dtype=np.float32)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    center = np.asarray(shape) // 2

    trunk = ((yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= 5**2) & (zz > 6) & (zz < 66)
    left_branch = (
        (yy - (center[1] - (zz - 38) * 0.55)) ** 2 + (xx - center[2]) ** 2 <= 3**2
    ) & (zz >= 38)
    right_branch = (
        (yy - (center[1] + (zz - 38) * 0.55)) ** 2 + (xx - center[2]) ** 2 <= 3**2
    ) & (zz >= 38)
    vessel = trunk | left_branch | right_branch
    hu[vessel] = 450.0

    # Add a low-contrast soft-tissue body around the vessel.
    body = (
        ((zz - center[0]) / 30.0) ** 2
        + ((yy - center[1]) / 27.0) ** 2
        + ((xx - center[2]) / 24.0) ** 2
        <= 1.0
    )
    hu[body & ~vessel] = 40.0
    return hu


def build_example(output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    ct_dir = output / "ct_cache"
    ct_dir.mkdir(parents=True, exist_ok=True)
    spacing = (1.0, 0.8, 0.8)

    hu = synthetic_contrast_ct()
    preprocessor = VolumePreprocessor.from_numpy(hu, spacing_zyx_mm=spacing)
    preprocessor.preprocess(output_dir=ct_dir)
    np.save(ct_dir / "hu_volume.npy", hu)

    segmentation = get_vessel_mask(
        hu_zyx=hu,
        spacing_zyx_mm=spacing,
        use_totalsegmentator=False,
        hu_threshold=200.0,
        min_component_voxels=100,
    )
    vessel_mask = segmentation.combined_mask
    np.save(ct_dir / "vessel_mask.npy", vessel_mask)

    vessel_result = convert_mask_to_usd(
        vessel_mask,
        output / "vasculature.usd",
        name="Vasculature",
        spacing_zyx_mm=spacing,
    )
    # Label 6 is the aorta in the package's standard anatomical category map.
    anatomy_result = convert_segmentation_array(
        vessel_mask.astype(np.uint8) * 6,
        output / "anatomy",
        spacing_zyx_mm=spacing,
    )

    manifest = {
        "ct_artifacts": [str(path) for path in sorted(ct_dir.iterdir())],
        "usd_files": [str(vessel_result.usd_path), str(anatomy_result.usd_path)],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return vessel_result.usd_path, anatomy_result.usd_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/vasculature_ct")
    args = parser.parse_args()
    for path in build_example(args.output_dir):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
