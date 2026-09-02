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
from i4h_asset_helper.assets import (
    get_i4h_asset_hash,
    get_i4h_asset_path,
    is_content_digest,
    sha256_of_folder,
    verify_asset,
)

# Catalog versions that publish a short S3 path segment instead of a content digest.
PATH_IDENTIFIER_VERSIONS = ("0.5.0", "0.6.0", "0.7.0")
CONTENT_DIGEST_VERSIONS = ("0.1.0ea", "0.1.0", "0.2.0", "0.3.0")

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


def test_manifest_records_path_identifiers_from_0_5_0_onwards():
    """The values are S3 path segments, so hashing the download can never reproduce them."""
    for version in PATH_IDENTIFIER_VERSIONS:
        value = get_i4h_asset_hash(version=version)
        assert not is_content_digest(value), version
        # The same value names the remote prefix the assets are served from.
        assert get_i4h_asset_path(version).endswith(f"/{version}/{value}")


def test_manifest_records_content_digests_before_0_5_0():
    for version in CONTENT_DIGEST_VERSIONS:
        assert is_content_digest(get_i4h_asset_hash(version=version)), version


def test_is_content_digest_rejects_short_and_non_hex_values():
    assert is_content_digest("a" * 64)
    assert is_content_digest("A" * 64)
    assert not is_content_digest("724f82e")
    assert not is_content_digest("a" * 63)
    assert not is_content_digest("z" * 64)


def test_verify_passes_for_complete_path_addressed_download(tmp_path, monkeypatch, capsys):
    """Regression test for NVBugs 6646790: this used to always fail on catalog 0.5.0+."""
    version = "0.7.0"
    _write_catalog(_local_dir(tmp_path, version), CATALOG_FILES)
    monkeypatch.setattr(assets, "_list_asset_entries", lambda url: _fake_listing(version, CATALOG_FILES))

    assert verify_asset(version=version, download_dir=str(tmp_path)) is True
    assert "Verification PASSED" in capsys.readouterr().out


def test_verify_reports_missing_file(tmp_path, monkeypatch, capsys):
    version = "0.7.0"
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


def test_verify_scopes_listing_to_sub_path(tmp_path, monkeypatch):
    """--verify after a --sub-path download must not demand the rest of the catalog."""
    version = "0.7.0"
    sub_path = "Props/Organs"
    organs = {k: v for k, v in CATALOG_FILES.items() if k.startswith(sub_path)}
    _write_catalog(_local_dir(tmp_path, version), organs)

    listed = {}

    def fake_listing(url):
        listed["url"] = url
        return _fake_listing(version, organs)

    monkeypatch.setattr(assets, "_list_asset_entries", fake_listing)

    assert verify_asset(version=version, download_dir=str(tmp_path), sub_path=sub_path) is True
    assert listed["url"].endswith(f"/{sub_path}")


def _staged_digest_download(tmp_path, files):
    """Lay out a download the way a content-digest catalog version addresses it."""
    staging = os.path.join(str(tmp_path), "staging")
    _write_catalog(staging, files)
    digest = sha256_of_folder(staging)
    local_dir = os.path.join(str(tmp_path), digest)
    os.rename(staging, local_dir)
    return digest, local_dir


def test_verify_hashes_content_for_digest_versions(tmp_path, capsys):
    digest, _ = _staged_digest_download(tmp_path, CATALOG_FILES)

    assert verify_asset(version="0.3.0", download_dir=str(tmp_path), hash=digest) is True
    assert "Verification PASSED" in capsys.readouterr().out


def test_verify_accepts_a_digest_given_in_uppercase(tmp_path, capsys):
    """Hex is case-insensitive, but sha256_of_folder only ever returns lowercase."""
    digest, local_dir = _staged_digest_download(tmp_path, CATALOG_FILES)
    os.rename(local_dir, os.path.join(str(tmp_path), digest.upper()))

    assert verify_asset(version="0.3.0", download_dir=str(tmp_path), hash=digest.upper()) is True
    assert "Verification PASSED" in capsys.readouterr().out


def test_verify_fails_when_digest_version_content_changes(tmp_path, capsys):
    digest, local_dir = _staged_digest_download(tmp_path, CATALOG_FILES)
    _write_catalog(local_dir, {"Robots/arm.usd": b"tampered"})

    assert verify_asset(version="0.3.0", download_dir=str(tmp_path), hash=digest) is False
    assert "hash mismatch" in capsys.readouterr().out


def test_verify_rejects_missing_download_dir(tmp_path):
    with pytest.raises(ValueError, match="Asset folder does not exist"):
        verify_asset(version="0.7.0", download_dir=str(tmp_path / "absent"))


def test_manifest_versions_are_covered_by_the_tests():
    """Guards against a new catalog version silently reintroducing the confusion."""
    manifest_path = os.path.join(os.path.dirname(assets.__file__), "assets_sha256.json")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    assert set(manifest) == set(PATH_IDENTIFIER_VERSIONS) | set(CONTENT_DIGEST_VERSIONS)
