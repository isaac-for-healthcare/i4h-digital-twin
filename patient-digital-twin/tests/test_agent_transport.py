# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK adapter contracts; model services are never called."""
import asyncio
import sys
from types import SimpleNamespace

import pytest


def test_codex_streams_actual_sdk_notification_shape(monkeypatch, tmp_path, capsys):
    sdk = pytest.importorskip('openai_codex')
    from patient_digital_twin.agent.__main__ import mcp_config, run_codex
    captured = {}

    class Turn:
        def stream(self):
            yield SimpleNamespace(method='item/agentMessage/delta', payload=SimpleNamespace(delta='done'))
            yield SimpleNamespace(method='turn/completed', payload=SimpleNamespace(turn=SimpleNamespace(status='completed')))

        def interrupt(self):
            captured['interrupted'] = True

    class FakeCodex:
        def __init__(self, config):
            captured['config'] = config

        def __enter__(self):
            return self

        def __exit__(self, *args):
            captured['closed'] = True

        def thread_start(self, **kwargs):
            captured['thread'] = kwargs
            return SimpleNamespace(turn=lambda prompt: Turn())

    monkeypatch.setattr(sdk, 'Codex', FakeCodex)
    monkeypatch.setenv('DISPLAY', ':77')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'never-forward-this')
    config = mcp_config()
    assert 'ANTHROPIC_API_KEY' not in config['env']
    run_codex('inspect anatomy', tmp_path, config, None)
    assert 'done' in capsys.readouterr().out
    assert captured['closed']
    assert 'interrupted' not in captured
    assert captured['thread']['sandbox'] == sdk.Sandbox.workspace_write
    assert any('mcp_servers.blender.command' in item for item in captured['config'].config_overrides)
    assert any('mcp_servers.blender.args=' in item for item in captured['config'].config_overrides)
    assert 'mcp_servers.blender.tools.look.approval_mode="approve"' in captured['config'].config_overrides
    assert not any('default_tools_approval_mode' in item for item in captured['config'].config_overrides)


def test_claude_receives_blender_tools(monkeypatch, tmp_path):
    sdk = pytest.importorskip('claude_agent_sdk')
    from patient_digital_twin.agent.__main__ import run_claude
    captured = {}

    async def query(**kwargs):
        captured.update(kwargs)
        yield sdk.ResultMessage(subtype='success', duration_ms=1, duration_api_ms=1,
                                is_error=False, num_turns=1, session_id='test')

    monkeypatch.setattr(sdk, 'query', query)
    asyncio.run(run_claude('inspect anatomy', tmp_path, {'command': sys.executable, 'args': []}, None, 3, 1.))
    options = captured['options']
    assert options.tools == []
    assert 'mcp__blender__execute_blender_code' in options.allowed_tools
    assert options.mcp_servers['blender']['command'] == sys.executable
    assert options.max_turns == 3 and options.max_budget_usd == 1.
