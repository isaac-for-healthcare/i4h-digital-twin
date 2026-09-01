# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the installable imaging_to_mesh API."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from pxr import Usd


def _sphere_mask(shape: tuple[int, int, int] = (24, 24, 24)) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return ((zz - 12) ** 2 + (yy - 12) ** 2 + (xx - 12) ** 2 <= 7**2).astype(np.uint8)


def test_public_exports():
    import imaging_to_mesh

    assert imaging_to_mesh.__version__ == "0.1.0"
    assert callable(imaging_to_mesh.convert_mask_to_usd)
    assert callable(imaging_to_mesh.convert_segmentation_file)
    assert callable(imaging_to_mesh.mask_to_mesh)


def test_mask_to_obj(tmp_path: Path):
    from imaging_to_mesh import mask_to_mesh, write_obj

    vertices, faces = mask_to_mesh(_sphere_mask(), spacing_zyx_mm=(1.0, 0.8, 0.8))
    obj_path = write_obj(vertices, faces, tmp_path / "vessel.obj")
    assert obj_path.is_file()
    assert len(vertices) > 0
    assert len(faces) > 0


def test_convert_numpy_mask_to_usd(tmp_path: Path):
    from imaging_to_mesh import convert_mask_to_usd

    result = convert_mask_to_usd(
        _sphere_mask(),
        tmp_path / "vessel.usd",
        name="Vasculature",
        spacing_zyx_mm=(1.0, 0.8, 0.8),
    )

    assert result.usd_path.is_file()
    assert len(result.meshes) == 1
    assert result.meshes[0].obj_path.is_file()
    assert result.meshes[0].vertex_count > 0
    assert result.meshes[0].face_count > 0
    stage = Usd.Stage.Open(str(result.usd_path))
    assert stage.GetPrimAtPath("/World/Vasculature").IsValid()


def test_convert_nifti_segmentation_to_usd(tmp_path: Path):
    from imaging_to_mesh import convert_segmentation_file

    labels_zyx = np.zeros((24, 24, 24), dtype=np.uint8)
    labels_zyx[_sphere_mask().astype(bool)] = 6  # Aorta, grouped under Veins
    labels_xyz = labels_zyx.transpose(2, 1, 0)
    source = tmp_path / "sample_label.nii.gz"
    nib.save(nib.Nifti1Image(labels_xyz, np.diag([0.8, 0.8, 1.0, 1.0])), source)

    result = convert_segmentation_file(source, tmp_path / "output")

    assert result.source_path == source
    assert result.usd_path == tmp_path / "output" / "all_organs.usd"
    assert result.usd_path.is_file()
    assert [mesh.name for mesh in result.meshes] == ["Veins"]
    assert (tmp_path / "output" / "obj" / "Veins.obj").is_file()


def test_vasculature_ct_to_usd_example(tmp_path: Path):
    pytest.importorskip("vasculature_digital_twin")
    example = Path(__file__).resolve().parents[1] / "examples" / "vasculature_ct_to_usd.py"
    spec = importlib.util.spec_from_file_location("vasculature_ct_to_usd", example)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    vessel_usd, anatomy_usd = module.build_example(tmp_path)

    assert vessel_usd.is_file()
    assert anatomy_usd.is_file()
    assert (tmp_path / "ct_cache" / "mu_volume.npy").is_file()
    assert (tmp_path / "manifest.json").is_file()
    stage = Usd.Stage.Open(str(vessel_usd))
    assert stage.GetPrimAtPath("/World/Vasculature").IsValid()
    anatomy = Usd.Stage.Open(str(anatomy_usd))
    assert anatomy.GetPrimAtPath("/World/Veins").IsValid()
