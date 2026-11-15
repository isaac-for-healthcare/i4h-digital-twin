# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse Blender or launch an owned GUI with the upstream MCP add-on."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

MCP_PACKAGE = 'mcp-for-blender==2.3.0'
BLENDER_VERSION = '5.1.2'
BLENDER_URL = f'https://download.blender.org/release/Blender5.1/blender-{BLENDER_VERSION}-linux-x64.tar.xz'
BLENDER_SHA256 = 'aaccb355f50183979b698bcce7467103a76261b5fa59f4972295842662a285fb'


def ensure_blender() -> str:
    """Use BLENDER_EXECUTABLE/PATH, otherwise cache the verified Linux x64 release."""
    requested = os.environ.get('BLENDER_EXECUTABLE')
    executable = shutil.which(requested or 'blender')
    if executable:
        return executable
    if requested:
        raise FileNotFoundError(f'BLENDER_EXECUTABLE does not exist: {requested}')
    if platform.system() != 'Linux' or platform.machine() not in {'x86_64', 'AMD64'}:
        raise RuntimeError('Automatic installation supports Linux x86_64; install Blender and set BLENDER_EXECUTABLE')
    cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'i4h/blender'
    destination = cache / f'blender-{BLENDER_VERSION}-linux-x64'
    if (destination / 'blender').is_file():
        return str(destination / 'blender')
    cache.mkdir(parents=True, exist_ok=True)
    print(f'Installing Blender {BLENDER_VERSION} in {cache}', file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix='.install-', dir=cache) as temp:
        archive = Path(temp) / 'blender.tar.xz'
        digest = hashlib.sha256()
        with urllib.request.urlopen(BLENDER_URL, timeout=60) as response, archive.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != BLENDER_SHA256:
            raise ValueError('Blender SHA-256 verification failed')
        unpacked = Path(temp) / 'unpacked'
        unpacked.mkdir()
        with tarfile.open(archive) as package:
            # Python 3.10 also needs traversal/link checks when data_filter is unavailable.
            for member in package.getmembers():
                target = (unpacked / member.name).resolve()
                if not target.is_relative_to(unpacked) or member.isdev() or member.isfifo():
                    raise ValueError('Unsafe path in Blender archive')
                if member.issym() or member.islnk():
                    link = ((target.parent if member.issym() else unpacked) / member.linkname).resolve()
                    if not link.is_relative_to(unpacked):
                        raise ValueError('Unsafe link in Blender archive')
            if hasattr(tarfile, 'data_filter'):
                package.extractall(unpacked, filter='data')
            else:
                package.extractall(unpacked)
        try:
            (unpacked / destination.name).rename(destination)
        except OSError:
            if not (destination / 'blender').is_file():
                raise
    return str(destination / 'blender')


def _responding(host: str, port: int) -> bool:
    """Read-only upstream capability handshake; a listening socket alone is insufficient."""
    try:
        with socket.create_connection((host, port), timeout=1) as connection:
            connection.sendall(b'{"type":"get_addon_info","params":{}}')
            data = bytearray()
            while len(data) < 1024 * 1024:
                chunk = connection.recv(65536)
                if not chunk:
                    return False
                data.extend(chunk)
                try:
                    response = json.loads(data)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(response, dict) or not isinstance(response.get('result'), dict):
                    return False
                return response.get('status') == 'success' and 'execute_code' in response['result'].get('capabilities', [])
    except OSError:
        pass
    return False


@contextlib.contextmanager
def blender_session(workspace: Path, config: dict, mode: str = 'auto'):
    """Yield MCP configuration; close only a Blender/Xvfb process group we launched."""
    if mode == 'existing':
        yield config
        return
    host = config.get('env', {}).get('BLENDER_HOST', '127.0.0.1')
    port = int(config.get('env', {}).get('BLENDER_PORT', '9876'))
    if mode == 'auto' and _responding(host, port):
        yield config
        return
    if mode == 'auto' and host not in {'localhost', '127.0.0.1', '::1'}:
        raise RuntimeError('Cannot start Blender on a remote host; start that instance yourself')
    virtual = mode == 'virtual' or (platform.system() == 'Linux' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))
    prefix = []
    if virtual:
        if platform.system() != 'Linux' or not shutil.which('xvfb-run') or not shutil.which('xauth'):
            raise RuntimeError('Virtual Blender requires Linux, xvfb-run and xauth (Ubuntu: sudo apt install xvfb xauth)')
        prefix = ['xvfb-run', '-a', '--server-args=-screen 0 1280x960x24 -nolisten tcp']
    executable = ensure_blender()
    addon = workspace / 'addon.py'
    # Load the add-on bundled with the same pinned upstream server; no user preferences are changed.
    subprocess.run(['uvx', '--from', MCP_PACKAGE, 'python', '-c',
                    'import shutil; from blender_mcp.addon_manager import get_bundled_addon_path; '
                    f'shutil.copyfile(get_bundled_addon_path(), {str(addon)!r})'], check=True, timeout=180)
    ready = workspace / 'blender-ready.json'
    ready.unlink(missing_ok=True)
    log_path = workspace / 'blender.log'
    env = {key: value for key, value in os.environ.items() if key not in {'PYTHONHOME', 'PYTHONPATH'}}
    env.update(BLENDERMCP_NO_UPDATE_CHECK='1', BLENDER_MCP_DISABLE_TELEMETRY='true')
    if virtual:
        env.pop('WAYLAND_DISPLAY', None)
        env.pop('WAYLAND_SOCKET', None)
    with log_path.open('w') as log:
        process = subprocess.Popen(prefix + [executable, '--factory-startup', '--python', str(Path(__file__).resolve()),
                                             '--', str(addon), str(ready)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=(os.name == 'posix'))
    blender_pid = None
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and process.poll() is None:
            if ready.is_file():
                info = json.loads(ready.read_text())
                port, blender_pid = info['port'], info['pid']
                if _responding('127.0.0.1', port):
                    yield {**config, 'env': {**config.get('env', {}), 'BLENDER_HOST': '127.0.0.1', 'BLENDER_PORT': str(port)}}
                    return
            time.sleep(.1)
        raise RuntimeError('Blender failed to become ready:\n' + log_path.read_text(errors='replace')[-4000:])
    finally:
        with contextlib.suppress(ProcessLookupError):
            if os.name != 'posix':
                process.terminate()
            elif blender_pid is not None:
                # Let xvfb-run reap Blender and remove its display before stopping the wrapper.
                os.kill(blender_pid, signal.SIGTERM)
            else:
                os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        finally:
            if os.name == 'posix':
                with contextlib.suppress(ProcessLookupError):
                    # xvfb-run signals Xvfb on exit but does not wait for its cleanup.
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        os.killpg(process.pid, 0)
                        time.sleep(.05)
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _start_addon() -> None:
    """Blender startup entry point: register upstream in this process, on a private port."""
    import importlib.util

    import bpy

    addon_path, ready_path = sys.argv[sys.argv.index('--') + 1:]
    spec = importlib.util.spec_from_file_location('i4h_blender_mcp', addon_path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load the upstream Blender MCP add-on')
    addon = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = addon
    spec.loader.exec_module(addon)
    addon.register()
    bpy.context.preferences.view.show_splash = False
    server = addon.BlenderMCPServer(host='127.0.0.1', port=0)
    bpy.types.blendermcp_server = server
    server.start()
    if not server.running:
        raise RuntimeError('Blender MCP add-on failed to start')
    ready = Path(ready_path)
    temporary = ready.with_suffix('.tmp')
    temporary.write_text(json.dumps({'port': server.socket.getsockname()[1], 'pid': os.getpid(), 'display': os.environ.get('DISPLAY')}))
    temporary.replace(ready)


if __name__ == '__main__':
    _start_addon()
