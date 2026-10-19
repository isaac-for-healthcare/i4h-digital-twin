# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Test the cli i4h-asset-retrieve."""

import importlib.util
import os
import subprocess
import sys
import tempfile

from sim_ready_assets.assets import _is_import_ready

CLI = [sys.executable, "-m", "sim_ready_assets.cli"]
# Pin the production S3 catalog even if the caller set I4H_ASSET_ENV.
PRODUCTION_ENV = {**os.environ, "I4H_ASSET_ENV": "production"}


def test_cli_retrieve():
    """Test the cli i4h-asset-retrieve."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = subprocess.run(
            [*CLI, "--version", "0.7.0", "--download-dir", temp_dir, "--sub-path", "Test"],
            check=True,
            capture_output=True,
            text=True,
            env=PRODUCTION_ENV,
        )
        assert "Assets downloaded to:" in result.stdout
        assert os.path.isdir(temp_dir)


def test_force_omni_client():
    """Test the cli i4h-asset-retrieve with force_omni_client."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [*CLI, "--version", "0.7.0", "--download-dir", temp_dir, "--sub-path", "Test", "--force_omni_client"]
        if importlib.util.find_spec("isaacsim") is None and not _is_import_ready("omni.client"):
            result = subprocess.run(cmd, capture_output=True, text=True, env=PRODUCTION_ENV)
            assert result.returncode != 0
            assert "Isaac Sim or omni.client is required" in result.stderr
            return

        subprocess.run(cmd, check=True, env=PRODUCTION_ENV)
