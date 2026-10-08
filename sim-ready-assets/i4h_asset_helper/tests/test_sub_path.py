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

"""Tests that --sub-path resolves files as well as folders (nvbugs 6893193). These do not touch the network."""

import sys

import pytest

from i4h_asset_helper import assets, cli
from i4h_asset_helper.assets import retrieve_asset

_BUCKET = "bucket"
_ROOT = "Assets/0.7.0"
_URL = f"https://{_BUCKET}.s3-us-west-2.amazonaws.com/{_ROOT}"
_KEYS = [
    f"{_ROOT}/Props/NuRec/nurec_orca_bg.usdz",
    f"{_ROOT}/Props/NuRec/nurec_or_bg.usdz",
    f"{_ROOT}/Robots/Franka/panda.usd",
]


class _FakeS3:
    """The two list_objects_v2 behaviours the helper relies on: prefix matching and delimiter grouping."""

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None, MaxKeys=1000):
        matches = sorted(key for key in _KEYS if key.startswith(Prefix))
        rests = [key[len(Prefix) :] for key in matches]
        prefixes = sorted({Prefix + rest.split("/")[0] + "/" for rest in rests if "/" in rest})
        contents = [Prefix + rest for rest in rests if "/" not in rest]
        response = {"KeyCount": min(len(contents) + len(prefixes), MaxKeys)}
        if contents:
            response["Contents"] = [{"Key": key, "Size": 1} for key in contents][:MaxKeys]
        if prefixes:
            response["CommonPrefixes"] = [{"Prefix": prefix} for prefix in prefixes]
        return response

    def get_paginator(self, name):
        class _Paginator:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": key, "Size": 1} for key in _KEYS if key.startswith(Prefix)]}

        return _Paginator()


@pytest.fixture
def fake_s3(monkeypatch):
    monkeypatch.setattr(assets, "_is_import_ready", lambda name: False)
    monkeypatch.setattr(assets, "_is_s3_environment", lambda: True)
    monkeypatch.setattr(assets, "_get_s3_client", lambda: _FakeS3())
    monkeypatch.setattr(assets, "get_i4h_asset_path", lambda version=None, hash=None: _URL)
    downloaded = []
    monkeypatch.setattr(assets, "_download_assets", lambda entries, *a, **k: downloaded.extend(entries))
    return downloaded


def test_a_file_key_is_not_a_folder(fake_s3):
    assert not assets._is_url_folder(f"{_URL}/Props/NuRec/nurec_orca_bg.usdz")
    assert assets._is_url_folder(f"{_URL}/Props/NuRec")


def test_a_partial_name_is_not_a_folder(fake_s3):
    # "Props/Nu" is a prefix of the "Props/NuRec/" folder but not a folder itself.
    assert not assets._is_url_folder(f"{_URL}/Props/Nu")


def test_retrieve_a_file_sub_path_downloads_that_file(tmp_path, fake_s3):
    retrieve_asset(version="0.7.0", download_dir=str(tmp_path), sub_path="Props/NuRec/nurec_orca_bg.usdz")

    assert [entry.rsplit("/", 1)[-1] for entry in fake_s3] == ["nurec_orca_bg.usdz"]


def test_retrieve_a_folder_sub_path_downloads_its_files(tmp_path, fake_s3):
    retrieve_asset(version="0.7.0", download_dir=str(tmp_path), sub_path="Props/NuRec")

    assert sorted(entry.rsplit("/", 1)[-1] for entry in fake_s3) == ["nurec_or_bg.usdz", "nurec_orca_bg.usdz"]


def test_retrieve_an_empty_listing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_list_asset_url", lambda url: [])

    with pytest.raises(FileNotFoundError, match="No assets match Props/Missing"):
        retrieve_asset(version="0.7.0", download_dir=str(tmp_path), sub_path="Props/Missing")


def test_cli_exits_non_zero_when_nothing_matches(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_is_s3_environment", lambda: True)
    monkeypatch.setattr(assets, "_list_asset_url", lambda url: [])
    monkeypatch.setattr(
        sys, "argv", ["i4h-asset-retrieve", "--download-dir", str(tmp_path), "--sub-path", "Props/Missing"]
    )

    assert cli.retrieve_main() == 1
    captured = capsys.readouterr()
    assert "No assets match Props/Missing" in captured.err
    assert "Assets downloaded to:" not in captured.out
