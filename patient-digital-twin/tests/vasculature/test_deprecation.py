# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vasculature_digital_twin package and its CLIs announce their deprecation."""

import subprocess
import sys

import pytest


def test_import_emits_deprecation_warning():
    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import vasculature_digital_twin"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "vasculature_digital_twin is deprecated" in result.stderr


def test_patient_package_does_not_import_the_deprecated_package():
    code = "import sys, patient_digital_twin; assert 'vasculature_digital_twin' not in sys.modules"
    subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c", code], check=True)


@pytest.mark.parametrize("module", ["preprocess_ct", "segment_vessels"])
def test_cli_warns_before_running(module):
    cli = __import__(f"vasculature_digital_twin.cli.{module}", fromlist=["main"])
    with pytest.warns(FutureWarning, match="deprecated"), pytest.raises(SystemExit):
        cli.main(["--help"])
