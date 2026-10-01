# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the documented APIs and syntax-check every fenced code example."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def blocks(path):
    return re.findall(r"^```([^\n]*)\n(.*?)^```", path.read_text(encoding="utf-8"), re.M | re.S)


@pytest.mark.parametrize("path", sorted(p for p in ROOT.rglob("*.md") if not any(part.startswith(".") or part in {"build", "dist"} for part in p.relative_to(ROOT).parts)), ids=lambda p: str(p.relative_to(ROOT)))
def test_all_documentation_code_syntax(path):
    for language, code in blocks(path):
        if language == "python":
            compile(code, str(path), "exec")
        elif language in {"bash", "sh"}:
            subprocess.run(["bash", "-n"], input=code, text=True, check=True)
        elif language == "json":
            json.loads(code)


def test_readme_apis_and_usd_inspection(tmp_path, monkeypatch):
    pytest.importorskip("pxr")
    pytest.importorskip("vtk")
    pytest.importorskip("SimpleITK")
    from test_scan_volume import write_dicom

    monkeypatch.chdir(tmp_path)
    (tmp_path / "patient-digital-twin").symlink_to(ROOT, target_is_directory=True)
    write_dicom(tmp_path / "dicom", angle=.3)
    namespace = {}
    for language, code in blocks(ROOT / "README.md"):
        if language == "python" and "NVSegmentImporter" not in code:
            exec(compile(code, "README.md", "exec"), namespace)
    for language, code in blocks(ROOT / "docs/usd.md"):
        if language == "python":
            exec(compile(code, "docs/usd.md", "exec"), namespace)


@pytest.mark.skipif(os.environ.get("PATIENT_TEST_MODELS") != "1", reason="set PATIENT_TEST_MODELS=1 with both model checkouts")
def test_documented_model_imports():
    # Execute the actual README block, substituting only user-supplied paths.
    code = next(code for language, code in blocks(ROOT / "README.md") if language == "python" and "NVSegmentImporter" in code)
    code = code.replace("/path/to/NV-Segment-CTMR/NV-Segment-CTMR", os.environ["NV_SEGMENT_BUNDLE"])
    code = code.replace("/path/to/NV-Generate-CTMR", os.environ["NV_GENERATE_ROOT"])
    namespace = {"sample": ROOT / "examples/data/s0011"}
    exec(compile(code, "README.md", "exec"), namespace)
    assert not namespace["anatomy"].structures["aorta"].is_empty
    assert namespace["generator"].ct_scan is not None
