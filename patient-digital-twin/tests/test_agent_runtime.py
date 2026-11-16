# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installer integrity, process ownership, and an opt-in real virtual desktop."""
import asyncio
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import pytest
from patient_digital_twin.agent import runtime


@pytest.mark.parametrize('valid', [True, False])
def test_installer_checks_hash_and_reuses_cached_binary(tmp_path, monkeypatch, valid):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:xz') as package:
        info = tarfile.TarInfo(f'blender-{runtime.BLENDER_VERSION}-linux-x64/blender')
        data = b'#!/bin/sh\nexit 0\n'
        info.size, info.mode = len(data), 0o755
        package.addfile(info, io.BytesIO(data))
    payload = archive.getvalue()
    monkeypatch.delenv('BLENDER_EXECUTABLE', raising=False)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    monkeypatch.setattr(runtime.shutil, 'which', lambda _: None)
    monkeypatch.setattr(runtime.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(runtime.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(runtime, 'BLENDER_SHA256', hashlib.sha256(payload).hexdigest() if valid else '0' * 64)
    monkeypatch.setattr(runtime.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(payload))
    if not valid:
        with pytest.raises(ValueError, match='SHA-256'):
            runtime.ensure_blender()
        assert not list(tmp_path.rglob('blender-*/blender'))
        return
    path = runtime.ensure_blender()
    assert Path(path).is_file() and os.access(path, os.X_OK)
    monkeypatch.setattr(runtime.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('Cached Blender was downloaded again'))
    assert runtime.ensure_blender() == path


def test_existing_blender_is_not_installed_or_owned(tmp_path, monkeypatch):
    config = {'command': 'unused', 'args': [], 'env': {'BLENDER_PORT': '9876'}}
    monkeypatch.setattr(runtime, '_responding', lambda *a: True)
    monkeypatch.setattr(runtime, 'ensure_blender', lambda: pytest.fail('Existing Blender must not trigger installation'))
    monkeypatch.setattr(runtime.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Existing Blender must not launch a process'))
    with pytest.raises(RuntimeError, match='agent failure'):
        with runtime.blender_session(tmp_path, config) as connection:
            assert connection is config
            raise RuntimeError('agent failure')


def test_real_virtual_desktop_and_cleanup_on_failure(tmp_path, monkeypatch):
    executable = os.environ.get('I4H_TEST_BLENDER')
    if not executable:
        pytest.skip('Set I4H_TEST_BLENDER and provide xvfb-run/xauth for the real runtime check')
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from patient_digital_twin.agent.__main__ import BLENDER_TOOLS, mcp_config

    monkeypatch.setenv('BLENDER_EXECUTABLE', executable)
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    config = mcp_config()
    config['env']['UV_CACHE_DIR'] = os.environ.get('UV_CACHE_DIR', '/tmp/i4h-uv-cache')
    with pytest.raises(RuntimeError, match='agent failure'):
        with runtime.blender_session(tmp_path, config, 'virtual') as connection:
            ready = json.loads((tmp_path / 'blender-ready.json').read_text())
            assert runtime._responding('127.0.0.1', ready['port'])
            with monkeypatch.context() as incompatible:
                incompatible.setattr(runtime, 'MCP_PROTOCOL', -1)
                assert not runtime._responding('127.0.0.1', ready['port'])
            display = Path('/tmp/.X11-unix/X' + ready['display'].split(':')[-1].split('.')[0])
            assert display.exists()
            with runtime.blender_session(tmp_path, connection) as reused:
                assert reused is connection
            assert runtime._responding('127.0.0.1', ready['port'])

            async def screenshot():
                async with stdio_client(StdioServerParameters(**connection)) as (read, write):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        assert set(BLENDER_TOOLS) <= {t.name for t in (await client.list_tools()).tools}
                        result = await client.call_tool('look', {'mode': 'viewport', 'shading': 'solid', 'max_size': 256})
                        assert not result.isError and any(c.type == 'image' for c in result.content)

            asyncio.run(screenshot())
            raise RuntimeError('agent failure')
    assert not runtime._responding('127.0.0.1', ready['port'])
    assert not display.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(ready['pid'], 0)
