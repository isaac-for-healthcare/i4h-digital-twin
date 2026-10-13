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

"""Verification tests for i4h-asset-retrieve --verify. These do not touch the network."""

import json
import os

import pytest

from i4h_asset_helper import assets
from i4h_asset_helper.assets import get_i4h_asset_hash, get_i4h_asset_path, verify_asset

# Catalog versions whose manifest entry is the catalog's short commit id.
COMMIT_ID_VERSIONS = ("0.5.0", "0.6.0", "0.7.0")
# Catalog versions whose manifest entry is the SHA-256 of the asset ZIP package, not of the served tree.
ZIP_DIGEST_VERSIONS = ("0.1.0ea", "0.1.0", "0.2.0", "0.3.0")

CATALOG_FILES = {
    "Props/Organs/organs.usd": b"organs",
    "Props/Organs/materials/skin.jpg": b"skin-texture",
    "Robots/arm.usd": b"arm",
}


@pytest.fixture(autouse=True)
def production_env(monkeypatch):
    monkeypatch.setenv("I4H_ASSET_ENV", "production")
    monkeypatch.delenv("I4H_ASSET_SHA256_HASH", raising=False)


def _write_catalog(root, files):
    for relative_path, payload in files.items():
        path = os.path.join(root, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(payload)


def _fake_listing(version, files):
    """Build what _list_asset_entries would return for the given catalog contents."""
    asset_root = get_i4h_asset_path(version)
    return [(f"{asset_root}/{relative_path}", len(payload)) for relative_path, payload in files.items()]


def _local_dir(tmp_path, version):
    return os.path.join(str(tmp_path), get_i4h_asset_hash(version=version))


@pytest.mark.parametrize("version", COMMIT_ID_VERSIONS + ZIP_DIGEST_VERSIONS)
def test_manifest_value_names_the_remote_prefix(version):
    """Every manifest value, digest-shaped or not, says where the assets live rather than what they contain."""
    value = get_i4h_asset_hash(version=version)
    assert get_i4h_asset_path(version).endswith(f"/{version}/{value}")


@pytest.mark.parametrize("version", COMMIT_ID_VERSIONS + ZIP_DIGEST_VERSIONS)
def test_verify_passes_for_complete_download(tmp_path, monkeypatch, capsys, version):
    """Regression test for NVBugs 6646790 (0.5.0+) and 6921218 (0.3.0 and earlier).

    Both used to always fail: the first hashed against a commit id, the second against the SHA-256 of a ZIP
    package the S3 catalog does not serve.
    """
    _write_catalog(_local_dir(tmp_path, version), CATALOG_FILES)
    monkeypatch.setattr(assets, "_list_asset_entries", lambda url: _fake_listing(version, CATALOG_FILES))

    assert verify_asset(version=version, download_dir=str(tmp_path)) is True
    output = capsys.readouterr().out
    assert "against catalog listing" in output
    assert "Verification PASSED" in output


@pytest.mark.parametrize("version", ("0.3.0", "0.7.0"))
def test_verify_reports_missing_file(tmp_path, monkeypatch, capsys, version):
    present = dict(CATALOG_FILES)
    dropped = present.pop("Robots/arm.usd")
    assert dropped
    _write_catalog(_local_dir(tmp_path, version), present)
    monkeypatch.setattr(assets, "_list_asset_entries", lambda url: _fake_listing(version, CATALOG_FILES))

    assert verify_asset(version=version, download_dir=str(tmp_path)) is False
    output = capsys.readouterr().out
    assert "Missing:       1" in output
    assert "missing: Robots/arm.usd" in output


def test_verify_reports_half_written_file(tmp_path, monkeypatch, capsys):
    """An interrupted or timed-out download leaves short files behind."""
    version = "0.7.0"
    truncated = dict(CATALOG_FILES)
    truncated["Props/Organs/organs.usd"] = b"org"
    _write_catalog(_local_dir(tmp_path, version), truncated)
    monkeypatch.setattr(assets, "_list_asset_entries", lambda url: _fake_listing(version, CATALOG_FILES))

    assert verify_asset(version=version, download_dir=str(tmp_path)) is False
    output = capsys.readouterr().out
    assert "Wrong size:    1" in output
    assert "wrong size: Props/Organs/organs.usd (catalog 6 B, local 3 B)" in output


@pytest.mark.parametrize("version", ("0.3.0", "0.7.0"))
def test_verify_scopes_listing_to_sub_path(tmp_path, monkeypatch, version):
    """--verify after a --sub-path download must not demand the rest of the catalog."""
    sub_path = "Props/Organs"
    organs = {k: v for k, v in CATALOG_FILES.items() if k.startswith(sub_path)}
    _write_catalog(_local_dir(tmp_path, version), organs)

    listed = {}

    def fake_listing(url):
        listed["url"] = url
        return _fake_listing(version, organs)

    monkeypatch.setattr(assets, "_list_asset_entries", fake_listing)

    assert verify_asset(version=version, download_dir=str(tmp_path), sub_path=sub_path) is True
    assert listed["url"] == f"{get_i4h_asset_path(version)}/{sub_path}"


def test_verify_rejects_missing_download_dir(tmp_path):
    with pytest.raises(ValueError, match="Asset folder does not exist"):
        verify_asset(version="0.7.0", download_dir=str(tmp_path / "absent"))


def test_manifest_versions_are_covered_by_the_tests():
    """Guards against a new catalog version silently reintroducing the confusion."""
    manifest_path = os.path.join(os.path.dirname(assets.__file__), "assets_sha256.json")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    assert set(manifest) == set(COMMIT_ID_VERSIONS) | set(ZIP_DIGEST_VERSIONS)
