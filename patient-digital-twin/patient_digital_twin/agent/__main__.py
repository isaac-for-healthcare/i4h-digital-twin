# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optionally edit pipeline assets through Claude/Codex and upstream Blender MCP."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import tempfile
from pathlib import Path

INSTRUCTIONS = """Edit the supplied I4H scene using Blender MCP. Inspect the scene and a
viewport screenshot before and after changes. Only edit this session's scene.
Keep the existing mesh objects and their mesh-data i4h_path properties: adding,
deleting, splitting or joining objects is unsupported. Geometry edits, modifiers,
UVs and USD Preview Surface compatible materials/textures are supported. Use
Principled BSDF and image textures; bake unsupported procedural shaders first.
Do not change scene units. Leave Object Mode active when finished. Do not export
or modify source files: the application validates and publishes the result.
"""


def mcp_config() -> dict:
    """Use the upstream server and the user's already-running Blender add-on."""
    return {"command": "uvx", "args": ["mcp-for-blender"],
            "env": {"BLENDER_MCP_DISABLE_TELEMETRY": "true",
                    **{key: os.environ[key] for key in ("BLENDER_HOST", "BLENDER_PORT") if key in os.environ}}}


async def blender_call(config: dict, function: str, workspace: Path) -> None:
    from datetime import timedelta

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    worker = Path(__file__).with_name('blender.py')
    code = f"exec(compile(open({str(worker)!r}).read(), {str(worker)!r}, 'exec')); {function}({str(workspace)!r})"
    async with stdio_client(StdioServerParameters(**config)) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=120)) as client:
            await client.initialize()
            result = await client.call_tool('execute_blender_code', {'code': code})
            marker = workspace / ('opened' if function == 'open_asset' else 'edited.usdc')
            if result.isError or not marker.is_file():
                raise RuntimeError(f'Blender {function} failed: {result.content}')


async def run_claude(prompt: str, workspace: Path, config: dict, model: str | None,
                     max_turns: int, budget_usd: float) -> None:
    """Stream a bounded Claude Agent SDK session using its normal authentication."""
    from typing import cast

    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
    from claude_agent_sdk.types import McpStdioServerConfig

    options = ClaudeAgentOptions(
        cwd=str(workspace), system_prompt=INSTRUCTIONS, model=model,
        tools=[], allowed_tools=["mcp__blender__" + name for name in ("execute_blender_code", "get_scene_info", "get_object_info", "get_viewport_screenshot")], mcp_servers={"blender": cast(McpStdioServerConfig, config)},
        setting_sources=[], max_turns=max_turns, max_budget_usd=budget_usd,
    )
    finished = False
    async for message in query(prompt=prompt, options=options):
        for block in getattr(message, "content", []):
            if text := getattr(block, "text", None):
                print(text, flush=True)
        if isinstance(message, ResultMessage):
            finished = True
            if message.is_error or message.subtype != "success":
                raise RuntimeError(f"Claude session ended: {message.subtype}")
    if not finished:
        raise RuntimeError("Claude session ended without a result")


def run_codex(prompt: str, workspace: Path, config: dict, model: str | None, timeout_s: float = 600) -> None:
    """Stream the Codex Python SDK; credentials/config remain owned by Codex."""
    from threading import Timer

    from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

    overrides = (
        f'mcp_servers.blender.command={json.dumps(config["command"])}',
        f'mcp_servers.blender.args={json.dumps(config["args"])}',
        *(f'mcp_servers.blender.env.{key}={json.dumps(value)}' for key, value in config.get('env', {}).items()),
    )
    # These are the capabilities exposed by this application (the same set
    # Claude permits through allowed_tools). Do not change user/global policy
    # or grant access to unrelated MCP servers or future unlisted tools.
    blender_tools = ('execute_blender_code', 'get_scene_info', 'get_object_info', 'get_viewport_screenshot')
    overrides += (f'mcp_servers.blender.enabled_tools={json.dumps(blender_tools)}',)
    overrides += tuple(f'mcp_servers.blender.tools.{name}.approval_mode="approve"' for name in blender_tools)
    with Codex(config=CodexConfig(cwd=str(workspace), config_overrides=overrides)) as codex:
        thread = codex.thread_start(cwd=str(workspace), model=model,
                                    developer_instructions=INSTRUCTIONS,
                                    sandbox=Sandbox.workspace_write, approval_mode=ApprovalMode.deny_all)
        turn = thread.turn(prompt)
        def interrupt():
            with contextlib.suppress(Exception):
                turn.interrupt()
        timer = Timer(timeout_s, interrupt)
        timer.daemon = True
        timer.start()
        completed = False
        try:
            for event in turn.stream():
                method = getattr(event, "method", "")
                params = getattr(event, "payload", None)
                if method == "item/agentMessage/delta":
                    print(getattr(params, "delta", ""), end="", flush=True)
                if method == "turn/completed":
                    result = getattr(params, "turn", None)
                    status = getattr(result, "status", None)
                    if getattr(status, "value", status) != "completed":
                        raise RuntimeError(f"Codex turn ended: {status}")
                    completed = True
        except BaseException:
            interrupt()
            raise
        finally:
            timer.cancel()
        if not completed:
            raise RuntimeError("Codex session ended without a completed turn")
        print()


def main(argv: list[str] | None = None) -> int:
    """Optionally edit an existing pipeline output; generation itself is unchanged."""
    from .assets import finalize, prepare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('claude', 'codex'), required=True)
    parser.add_argument('--input', type=Path, required=True, help='Pipeline USD or patient_twin.yaml')
    parser.add_argument('--output', type=Path, required=True, help='New final asset directory')
    parser.add_argument('--mcp-config', type=Path, help='JSON stdio server config overriding upstream uvx default')
    parser.add_argument('--vessel-names', nargs='+', help='Vessels used to regenerate existing navigation arrays')
    parser.add_argument('--model')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--max-turns', type=int, default=30)
    parser.add_argument('--budget-usd', type=float, default=5)
    parser.add_argument('prompt')
    args = parser.parse_args(argv)
    if args.max_turns <= 0 or not 0 < args.budget_usd < float('inf') or not 0 < args.timeout < float('inf'):
        parser.error('max-turns, budget-usd and timeout must be positive and finite')
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('Use a new output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.mcp_config.read_text()) if args.mcp_config else mcp_config()
    # Same host/filesystem as Blender; the temporary files never replace the input.
    with tempfile.TemporaryDirectory(prefix='.asset-edit-', dir=output.parent) as temp:
        workspace = Path(temp)
        settings = prepare(args.input, workspace)
        asyncio.run(blender_call(config, 'open_asset', workspace))
        prompt = f"Edit scene I4H {settings['token']}.\n{args.prompt}"
        if args.backend == 'claude':
            async def bounded():
                await asyncio.wait_for(run_claude(prompt, workspace, config, args.model,
                                                 args.max_turns, args.budget_usd), args.timeout)
            asyncio.run(bounded())
        else:
            run_codex(prompt, workspace, config, args.model, args.timeout)
        asyncio.run(blender_call(config, 'export_asset', workspace))
        print(finalize(workspace, output, vessel_names=args.vessel_names))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
