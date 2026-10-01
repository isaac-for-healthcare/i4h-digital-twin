# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process model adapters must not confuse upstream imports or retain CLI state."""

import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from patient_digital_twin.importers._common import backend_context, runtime


@pytest.mark.parametrize("fail", [False, True])
def test_backend_switching_restores_host_state(tmp_path, monkeypatch, fail):
    host = ModuleType("scripts")
    child = ModuleType("scripts.host")
    monkeypatch.setitem(sys.modules, "scripts", host)
    monkeypatch.setitem(sys.modules, "scripts.host", child)
    cwd, path, argv = Path.cwd(), sys.path[:], sys.argv
    for name in ("segment", "generate", "segment"):
        root = tmp_path / name
        (root / "scripts").mkdir(parents=True, exist_ok=True)
        (root / "scripts/__init__.py").write_text("")
        (root / "scripts/model.py").write_text(f"BACKEND = {name!r}\n")
        try:
            with backend_context(root):
                assert importlib.import_module("scripts.model").BACKEND == name
                assert "scripts.host" not in sys.modules
                assert Path.cwd() == root
                sys.argv = ["model", "--example"]
                if fail:
                    raise RuntimeError("inference failed")
        except RuntimeError:
            assert fail
        assert Path.cwd() == cwd and sys.path == path and sys.argv is argv
        assert sys.modules["scripts"] is host
        assert sys.modules["scripts.host"] is child
        assert "scripts.model" not in sys.modules


def test_default_runtime_does_not_launch_python_and_has_install_hint(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("subprocess"))
    assert runtime(None, ["json"], "patient-digital-twin[nvsegment]") is None
    with pytest.raises(ImportError, match=r"patient-digital-twin\[nvsegment\]"):
        runtime(None, ["not_an_installed_inference_dependency"], "patient-digital-twin[nvsegment]")


def test_base_import_does_not_require_inference_dependencies():
    result = subprocess.run(
        [sys.executable, "-B", "-c", """
import sys
import importlib.abc
class NoInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'monai', 'ignite', 'einops', 'huggingface_hub'}:
            raise AssertionError('Optional inference imported: ' + fullname)
sys.meta_path.insert(0, NoInference())
from patient_digital_twin import HumanBody, NVGenerateImporter, NVSegmentImporter
"""],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
