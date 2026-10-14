# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default dev Nucleus catalog through the standalone omni.client. These do not touch the network."""

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from sim_ready_assets import assets
from sim_ready_assets.assets import (
    get_i4h_local_asset_path,
    retrieve_asset,
    verify_asset,
)

NUCLEUS_ROOT = "omniverse://isaac-dev.ov.nvidia.com/Library/IsaacHealthcare/v0.9.0"
CATALOG_FILES = {
    "Props/Organs/organs.usd": b"organs",
    "Props/Organs/materials/organs/Bladder/LICENSE": b"license",
    "Robots/arm.usd": b"arm",
}
FOLDER, FILE = 1, 2


@pytest.fixture
def nucleus(monkeypatch):
    """A fake omni.client serving CATALOG_FILES, with Isaac Sim unavailable."""

    def children(url):
        prefix = url.rstrip("/")[len(NUCLEUS_ROOT) :].strip("/")
        prefix = f"{prefix}/" if prefix else ""
        found = {}
        for path, payload in CATALOG_FILES.items():
            if path.startswith(prefix):
                name, _, rest = path[len(prefix) :].partition("/")
                found[name] = (FOLDER, 0) if rest else (FILE, len(payload))
        return [SimpleNamespace(relative_path=name, flags=flags, size=size) for name, (flags, size) in found.items()]

    def read_file(url):
        return "OK", None, CATALOG_FILES[url[len(NUCLEUS_ROOT) + 1 :]]

    client = ModuleType("omni.client")
    client.Result = SimpleNamespace(OK="OK")
    client.ItemFlags = SimpleNamespace(CAN_HAVE_CHILDREN=FOLDER, READABLE_FILE=FILE)
    client.list = lambda url: ("OK", children(url))
    client.stat = lambda url: ("OK", SimpleNamespace(size=0))
    client.read_file = read_file
    omni = ModuleType("omni")
    omni.client = client
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.client", client)
    monkeypatch.setattr(assets, "_is_import_ready", lambda name: name == "omni.client")
    for name in ("I4H_ASSET_ENV", "I4H_ASSET_VERSION", "I4H_ASSET_SHA256_HASH"):
        monkeypatch.delenv(name, raising=False)


def test_default_catalog_is_listed_without_isaac_sim(nucleus):
    entries = assets._list_asset_entries(assets.get_i4h_asset_path())

    assert sorted(entries) == sorted((f"{NUCLEUS_ROOT}/{path}", len(data)) for path, data in CATALOG_FILES.items())


def test_default_catalog_downloads_and_verifies_without_isaac_sim(nucleus, tmp_path):
    local_dir = retrieve_asset(download_dir=str(tmp_path), concurrency=1)

    assert local_dir == get_i4h_local_asset_path(download_dir=str(tmp_path)) == str(tmp_path / "v0.9.0")
    for path, payload in CATALOG_FILES.items():
        with open(os.path.join(local_dir, path), "rb") as handle:
            assert handle.read() == payload
    assert verify_asset(download_dir=str(tmp_path))

    os.remove(os.path.join(local_dir, "Robots/arm.usd"))
    assert not verify_asset(download_dir=str(tmp_path))
