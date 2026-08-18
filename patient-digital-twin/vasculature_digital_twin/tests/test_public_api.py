# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API tests that produce the README-listed CT / vessel artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

README_ARTIFACTS = (
    "mu_volume.npy",
    "metadata.json",
    "hu_volume.npy",
    "vessel_mask.npy",
    "centerline_points_mm.npy",
    "centerline_edges.npy",
    "centerline_radii_mm.npy",
)


def test_package_exports():
    import vasculature_digital_twin as vdt

    assert isinstance(vdt.__version__, str)
    for name in (
        "VolumePreprocessor",
        "PreprocessedVolume",
        "PreprocessingSettings",
        "get_vessel_mask",
        "vessel_mask_from_hu",
        "extract_vessel_mesh",
    ):
        assert hasattr(vdt, name)


def test_readme_artifacts_from_synthetic_ct(tmp_path: Path, synthetic_ct_hu, spacing_zyx_mm):
    """Confirm README output artifacts can be extracted from a CT volume."""
    from vasculature_digital_twin import VolumePreprocessor, get_vessel_mask
    from vasculature_digital_twin.cli.segment_vessels import (
        main as segment_vessels_main,
    )

    ct_dir = tmp_path / "ct_cache"
    ct_dir.mkdir()

    # README preprocess path: HU volume -> mu_volume.npy + metadata.json
    preprocessor = VolumePreprocessor.from_numpy(
        synthetic_ct_hu,
        spacing_zyx_mm=spacing_zyx_mm,
    )
    volume = preprocessor.preprocess(output_dir=ct_dir)
    assert volume.shape == synthetic_ct_hu.shape
    assert (ct_dir / "mu_volume.npy").is_file()
    assert (ct_dir / "metadata.json").is_file()

    # README/CLI --save-hu path
    np.save(ct_dir / "hu_volume.npy", preprocessor.hu_volume_zyx.astype(np.float32))

    # Public vessel API without TotalSegmentator (CPU-friendly HU threshold path).
    result = get_vessel_mask(
        hu_zyx=preprocessor.hu_volume_zyx,
        spacing_zyx_mm=spacing_zyx_mm,
        use_totalsegmentator=False,
        hu_threshold=200.0,
        min_component_voxels=50,
    )
    assert result.combined_mask.shape == synthetic_ct_hu.shape
    assert int(result.combined_mask.sum()) > 0

    # README segment CLI writes vessel mask + centerline graph artifacts.
    rc = segment_vessels_main(
        [
            "--ct-dir",
            str(ct_dir),
            "--no-totalsegmentator",
            "--hu-threshold",
            "200",
            "--min-component-voxels",
            "50",
            "--close-iterations",
            "0",
        ]
    )
    assert rc == 0

    for name in README_ARTIFACTS:
        path = ct_dir / name
        assert path.is_file(), f"missing README artifact: {name}"

    meta = json.loads((ct_dir / "metadata.json").read_text(encoding="utf-8"))
    assert tuple(meta["spacing_zyx_mm"]) == spacing_zyx_mm
    assert tuple(meta["shape_zyx"]) == synthetic_ct_hu.shape

    mu = np.load(ct_dir / "mu_volume.npy")
    hu = np.load(ct_dir / "hu_volume.npy")
    vessel_mask = np.load(ct_dir / "vessel_mask.npy")
    pts_mm = np.load(ct_dir / "centerline_points_mm.npy")
    edges = np.load(ct_dir / "centerline_edges.npy")
    radii_mm = np.load(ct_dir / "centerline_radii_mm.npy")

    assert mu.shape == synthetic_ct_hu.shape
    assert hu.shape == synthetic_ct_hu.shape
    assert vessel_mask.shape == synthetic_ct_hu.shape
    assert vessel_mask.dtype == np.uint8
    assert int(vessel_mask.sum()) > 0
    assert pts_mm.ndim == 2 and pts_mm.shape[1] == 3 and pts_mm.shape[0] >= 2
    assert edges.ndim == 2 and edges.shape[1] == 2 and edges.shape[0] >= 1
    assert radii_mm.shape == (pts_mm.shape[0],)
    assert np.isfinite(pts_mm).all()
    assert np.isfinite(radii_mm).all()


def test_readme_handoff_loads_centerline_and_mask(tmp_path: Path, synthetic_ct_hu, spacing_zyx_mm):
    """Exercise the README handoff snippet against generated artifacts."""
    from vasculature_digital_twin import VolumePreprocessor, get_vessel_mask
    from vasculature_digital_twin.cli.segment_vessels import (
        main as segment_vessels_main,
    )

    ct_dir = tmp_path / "ct_cache"
    ct_dir.mkdir()
    pre = VolumePreprocessor.from_numpy(synthetic_ct_hu, spacing_zyx_mm=spacing_zyx_mm)
    pre.preprocess(output_dir=ct_dir)
    np.save(ct_dir / "hu_volume.npy", pre.hu_volume_zyx.astype(np.float32))
    assert (
        segment_vessels_main(
            [
                "--ct-dir",
                str(ct_dir),
                "--no-totalsegmentator",
                "--hu-threshold",
                "200",
                "--min-component-voxels",
                "50",
                "--close-iterations",
                "0",
            ]
        )
        == 0
    )

    meta = json.loads((ct_dir / "metadata.json").read_text(encoding="utf-8"))
    spacing = tuple(meta["spacing_zyx_mm"])
    origin_xyz_mm = tuple(meta.get("origin_xyz_mm") or (0.0, 0.0, 0.0))
    pts_mm = np.load(ct_dir / "centerline_points_mm.npy")
    vessel_mask = np.load(ct_dir / "vessel_mask.npy")

    track_start = pts_mm[0] / 1000.0
    track_dir = pts_mm[1] - pts_mm[0]
    track_dir = track_dir / (np.linalg.norm(track_dir) + 1e-12)
    track_length = float(np.linalg.norm((pts_mm[-1] - pts_mm[0]) / 1000.0))

    assert track_start.shape == (3,)
    assert abs(float(np.linalg.norm(track_dir)) - 1.0) < 1e-5
    assert track_length >= 0.0
    assert vessel_mask.shape == synthetic_ct_hu.shape
    assert spacing == spacing_zyx_mm
    assert origin_xyz_mm == (0.0, 0.0, 0.0)

    # Mesh extraction is optional (vtk/warp). Verify the public import surface.
    from vasculature_digital_twin.vasculature import extract_vessel_mesh

    assert callable(extract_vessel_mesh)
    # Keep get_vessel_mask exercised as the public segmentation entry point.
    assert get_vessel_mask is not None
