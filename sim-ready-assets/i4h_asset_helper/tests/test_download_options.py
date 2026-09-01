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

"""Tests that the download budget is reachable from the CLI. These do not touch the network."""

import sys
import threading

import pytest

from i4h_asset_helper import assets, cli
from i4h_asset_helper.assets import (
    DEFAULT_DOWNLOAD_CONCURRENCY,
    DEFAULT_DOWNLOAD_TIMEOUT,
    retrieve_asset,
)


def _capture_download(monkeypatch):
    """Stub out listing and downloading, recording the keyword arguments used."""
    recorded = {}

    def fake_download(url_entries, download_dir, version=None, hash=None, **kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(assets, "_list_asset_url", lambda url: [f"{url}/organs.usd"])
    monkeypatch.setattr(assets, "_filter_downloaded_assets", lambda entries, *a, **k: entries)
    monkeypatch.setattr(assets, "_download_assets", fake_download)
    return recorded


def test_retrieve_asset_defaults_match_the_module_defaults(tmp_path, monkeypatch):
    recorded = _capture_download(monkeypatch)

    retrieve_asset(version="0.7.0", download_dir=str(tmp_path))

    assert recorded == {
        "concurrency": DEFAULT_DOWNLOAD_CONCURRENCY,
        "timeout": DEFAULT_DOWNLOAD_TIMEOUT,
    }


def test_retrieve_asset_forwards_an_overridden_budget(tmp_path, monkeypatch):
    recorded = _capture_download(monkeypatch)

    retrieve_asset(version="0.7.0", download_dir=str(tmp_path), concurrency=8, timeout=18000.0)

    assert recorded == {"concurrency": 8, "timeout": 18000.0}


def test_download_timeout_cancels_the_queued_downloads(tmp_path, monkeypatch):
    """as_completed's timeout only ends the wait, so the queue has to be cancelled to stop the transfer."""
    started = []
    release = threading.Event()

    def fake_download(url_entry, download_dir, version=None, hash=None):
        started.append(url_entry)
        # Outlasts the timeout below, but bounded so a regression here fails rather than hangs.
        release.wait(1.0)
        return url_entry

    monkeypatch.setattr(assets, "_download_individual_asset", fake_download)
    url_entries = [f"s3://bucket/{index}.usd" for index in range(8)]

    try:
        with pytest.raises(TimeoutError, match="budget ran out"):
            assets._download_assets(url_entries, str(tmp_path), version="0.7.0", concurrency=1, timeout=0.2)
    finally:
        release.set()

    # Left queued, the pool would drain every entry before _download_assets could return.
    assert len(started) < len(url_entries)


def test_cli_forwards_the_download_budget_and_sub_path(tmp_path, monkeypatch):
    """A full catalog fetch can outlast the default hour, so --timeout has to reach the downloader."""
    recorded = {}

    def fake_retrieve(**kwargs):
        recorded["retrieve"] = kwargs
        return str(tmp_path)

    def fake_verify(**kwargs):
        recorded["verify"] = kwargs
        return True

    monkeypatch.setattr(cli, "retrieve_asset", fake_retrieve)
    monkeypatch.setattr(cli, "verify_asset", fake_verify)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "i4h-asset-retrieve",
            "--version",
            "0.7.0",
            "--download-dir",
            str(tmp_path),
            "--sub-path",
            "Props/Organs",
            "--concurrency",
            "6",
            "--timeout",
            "18000",
            "--verify",
        ],
    )

    assert cli.retrieve_main() == 0
    assert recorded["retrieve"]["concurrency"] == 6
    assert recorded["retrieve"]["timeout"] == 18000.0
    assert recorded["retrieve"]["sub_path"] == "Props/Organs"
    # Verification must cover the sub path that was downloaded, not the whole catalog.
    assert recorded["verify"]["sub_path"] == "Props/Organs"


def test_cli_reports_failure_when_verification_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "retrieve_asset", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(cli, "verify_asset", lambda **kwargs: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["i4h-asset-retrieve", "--version", "0.7.0", "--download-dir", str(tmp_path), "--verify"],
    )

    assert cli.retrieve_main() == 1
