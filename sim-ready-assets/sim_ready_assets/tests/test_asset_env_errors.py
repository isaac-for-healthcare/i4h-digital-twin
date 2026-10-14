# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Tests for the error raised when an S3 asset server does not have a file. These do not touch the network."""

import pytest
from botocore.exceptions import ClientError

from sim_ready_assets import assets

URL = "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.7.0/724f82e/Robots/Franka"


class _FailingS3Client:
    def __init__(self, code):
        self.code = code

    def download_file(self, bucket, key, local_path):
        raise ClientError({"Error": {"Code": self.code, "Message": "Not Found"}}, "HeadObject")


def _download_failing_with(code, env, tmp_path, monkeypatch):
    monkeypatch.setenv("I4H_ASSET_ENV", env)
    monkeypatch.setattr(assets, "_is_import_ready", lambda name: False)
    monkeypatch.setattr(assets, "_get_s3_client", lambda: _FailingS3Client(code))
    with pytest.raises(ValueError) as excinfo:
        assets._download_individual_asset(URL, str(tmp_path), version="0.7.0", hash="724f82e")
    return str(excinfo.value)


def test_not_found_on_staging_points_at_production(tmp_path, monkeypatch):
    message = _download_failing_with("404", "staging", tmp_path, monkeypatch)

    assert message.startswith(f"Failed to download asset from S3: {URL}")
    assert "'staging' asset server may not host this catalog version" in message
    assert "production" in message


def test_not_found_on_production_has_no_env_hint(tmp_path, monkeypatch):
    message = _download_failing_with("404", "production", tmp_path, monkeypatch)

    assert "may not host this catalog version" not in message


def test_other_errors_on_staging_have_no_env_hint(tmp_path, monkeypatch):
    message = _download_failing_with("403", "staging", tmp_path, monkeypatch)

    assert "may not host this catalog version" not in message
