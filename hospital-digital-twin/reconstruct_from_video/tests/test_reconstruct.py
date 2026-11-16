# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reconstruct.py"
spec = importlib.util.spec_from_file_location("reconstruct", SCRIPT)
reconstruct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconstruct)


def write_model(model_dir, images, points):
    model_dir.mkdir(parents=True)
    (model_dir / "images.bin").write_bytes(struct.pack("<Q", images))
    (model_dir / "points3D.bin").write_bytes(struct.pack("<Q", points))


class SelectLargestModelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sparse = Path(self.temp.name) / "sparse"

    def size(self, name):
        return reconstruct._model_size(self.sparse / name)

    def test_fragment_in_sparse0_is_swapped_out(self):
        write_model(self.sparse / "0", 3, 12)
        write_model(self.sparse / "1", 600, 400000)
        reconstruct.select_largest_model(self.sparse)
        self.assertEqual(self.size("0"), (600, 400000))
        self.assertEqual(self.size("1"), (3, 12))

    def test_largest_among_three_moves_to_sparse0(self):
        write_model(self.sparse / "0", 10, 50)
        write_model(self.sparse / "1", 20, 90)
        write_model(self.sparse / "2", 500, 300000)
        reconstruct.select_largest_model(self.sparse)
        self.assertEqual(self.size("0"), (500, 300000))
        self.assertEqual(self.size("2"), (10, 50))
        self.assertEqual(self.size("1"), (20, 90))

    def test_single_or_already_largest_is_unchanged(self):
        write_model(self.sparse / "0", 600, 400000)
        reconstruct.select_largest_model(self.sparse)
        self.assertEqual(self.size("0"), (600, 400000))
        write_model(self.sparse / "1", 3, 12)
        reconstruct.select_largest_model(self.sparse)
        self.assertEqual(self.size("0"), (600, 400000))
        self.assertEqual(self.size("1"), (3, 12))

    def test_root_owned_sparse_is_swapped_in_the_colmap_image(self):
        write_model(self.sparse / "0", 3, 12)
        write_model(self.sparse / "1", 600, 400000)
        with patch.object(Path, "rename", side_effect=PermissionError), \
                patch.object(reconstruct.subprocess, "run") as run:
            reconstruct.select_largest_model(self.sparse)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:4], ["docker", "run", "--rm", "-v"])
        self.assertEqual(cmd[4], f"{self.sparse.resolve()}:/sparse")
        self.assertEqual(cmd[5], reconstruct.COLMAP_IMAGE)
        self.assertEqual(cmd[-1], "mv /sparse/0 /sparse/_swap && mv /sparse/1 /sparse/0 && mv /sparse/_swap /sparse/1")
        self.assertTrue(run.call_args.kwargs["check"])

    def test_reused_work_dir_is_repaired_before_training(self):
        root = Path(self.temp.name) / "nurec"
        (root / "scripts").mkdir(parents=True)
        (root / "3dgrut").mkdir()
        (root / "3dgrut/install_env.sh").touch()
        write_model(root / "colmap/sparse/0", 3, 12)
        write_model(root / "colmap/sparse/1", 600, 400000)
        with patch.object(reconstruct, "__file__", str(root / "scripts/reconstruct.py")), \
                patch("sys.argv", [str(SCRIPT), "--work-dir", str(root)]), \
                patch.object(reconstruct.subprocess, "run"):
            self.assertEqual(reconstruct.main(), 0)
        self.assertEqual(reconstruct._model_size(root / "colmap/sparse/0"), (600, 400000))


class ReconstructionBuildTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "3dgrut").mkdir()
        (self.root / "3dgrut/install_env.sh").touch()
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
            "-t", "i4h-3dgrut:cu128", str(self.root / "3dgrut"),
        ])
        training = run.call_args_list[1].args[0]
        self.assertIn("i4h-3dgrut:cu128", training)
        self.assertIn("export_usdz.enabled=true", training)

    def test_failed_build_prevents_training(self):
        error = subprocess.CalledProcessError(1, ["docker", "build"])
        with patch.object(reconstruct.subprocess, "run", side_effect=error) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                reconstruct.main()
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
