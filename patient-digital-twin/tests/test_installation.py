# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression against source-path workarounds: use isolated installed imports."""

import subprocess
import sys
from pathlib import Path


def test_installed_meshing_from_unrelated_directory(tmp_path):
    # -I ignores environment path overrides and excludes the current directory.
    code = """
import sys
import numpy as np
from patient_digital_twin import SegmentationImporter
from patient_digital_twin.geometry import mask_to_mesh
from patient_digital_twin.export import export_to_usd, export_patient_twin
assert all(callable(f) for f in (export_to_usd, export_patient_twin))
vertices, faces = mask_to_mesh(np.ones((3, 3, 3)))
assert len(vertices) and len(faces)
body = SegmentationImporter(np.ones((3, 3, 3)), {1: "liver"}).to_anatomy_collection()
assert not body.structures["liver"].is_empty
assert not {"pxr", "trimesh", "vtk"} & sys.modules.keys()
print("installed patient mesher OK")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "installed patient mesher OK" in result.stdout


def test_pipeline_help_uses_installed_package(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "pipeline.py"
    result = subprocess.run(
        [sys.executable, "-I", str(example), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--source" in result.stdout and "--labels" in result.stdout
