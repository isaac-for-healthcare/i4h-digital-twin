# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reconstruct.py"
spec = importlib.util.spec_from_file_location("reconstruct", SCRIPT)
reconstruct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconstruct)


class ReconstructionBuildTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "3dgrut").mkdir()
        (self.root / "3dgrut/install_env_uv.sh").touch()
        (self.root / "colmap/sparse/0").mkdir(parents=True)
        self.file_patch = patch.object(reconstruct, "__file__", str(self.root / "scripts/reconstruct.py"))
        self.file_patch.start()
        self.addCleanup(self.file_patch.stop)
        self.argv_patch = patch("sys.argv", [str(SCRIPT), "--work-dir", str(self.root)])
        self.argv_patch.start()
        self.addCleanup(self.argv_patch.stop)

    def test_build_uses_owned_dockerfile_and_upstream_context(self):
        with patch.object(reconstruct.subprocess, "run") as run:
            self.assertEqual(reconstruct.main(), 0)
        build = run.call_args_list[0].args[0]
        self.assertEqual(build, [
            "docker", "build", "-f", str(self.root / "Dockerfile.3dgrut"),
            "-t", "i4h-3dgrut:cu130", str(self.root / "3dgrut"),
        ])
        training = run.call_args_list[1].args[0]
        self.assertIn("i4h-3dgrut:cu130", training)
        self.assertIn("/workspace/.venv/bin/python", training)
        self.assertIn("export_usdz.enabled=true", training)

    def test_x86_uses_the_published_colmap_image(self):
        (self.root / "colmap/sparse/0").rmdir()
        (self.root / "images").mkdir()
        with patch.object(reconstruct.platform, "machine", return_value="x86_64"), \
                patch.object(reconstruct.subprocess, "run") as run:
            self.assertEqual(reconstruct.main(), 0)
        colmap = run.call_args_list[0].args[0]
        self.assertIn("colmap/colmap:20251107.4118", colmap)

    def test_aarch64_builds_colmap_before_running_it(self):
        (self.root / "colmap/sparse/0").rmdir()
        (self.root / "images").mkdir()
        (self.root / "Dockerfile.colmap").write_text("FROM scratch\n")
        with patch.object(reconstruct.platform, "machine", return_value="aarch64"), \
                patch.object(reconstruct.subprocess, "run") as run:
            self.assertEqual(reconstruct.main(), 0)
        build, colmap = run.call_args_list[0].args[0], run.call_args_list[1].args[0]
        self.assertEqual(build, ["docker", "build", "-t", "i4h-colmap:3.13.0", "-"])
        self.assertIn("i4h-colmap:3.13.0", colmap)
        self.assertNotIn("colmap/colmap:20251107.4118", colmap)

    def test_failed_build_prevents_training(self):
        error = subprocess.CalledProcessError(1, ["docker", "build"])
        with patch.object(reconstruct.subprocess, "run", side_effect=error) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                reconstruct.main()
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
