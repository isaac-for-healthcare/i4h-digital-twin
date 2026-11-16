# Optional agent editing

Generate a patient USD or bundle normally, then optionally edit it with
Claude/Codex and [Blender MCP](https://github.com/ahujasid/blender-mcp).

From the repository root:

```bash
uv pip install -e './patient-digital-twin[pipeline,codex]'
```

Authenticate Codex, then run (Blender starts automatically if needed):

```bash
patient-digital-twin-agent --backend codex \
  --input output/patient_twin.yaml --output edited-patient \
  'Smooth the liver surface and give it a red material. Inspect the result.'
```

The default reuses a compatible local Blender MCP session, or launches Blender
with the upstream add-on. With no display on Linux it uses Xvfb. To always use a
private virtual desktop, add `--blender-mode virtual` (Ubuntu prerequisites:
`sudo apt install xvfb xauth`). This runs a GUI on a virtual display so viewport
screenshots work; it is not Blender's background mode or a VNC server.

If Blender is missing on Linux x86_64, the verified official 5.1.2 archive is
installed under `$XDG_CACHE_HOME/i4h/blender` (default `~/.cache/i4h/blender`).
Set `BLENDER_EXECUTABLE` to select an installation; other platforms need Blender
on PATH or this override. Managed startup is tested on Linux.
The matching MCP server/add-on is pinned to 2.3.0.
Only processes started by the command are closed on completion or failure;
existing Blender sessions and preferences are left intact. `--blender-mode existing`
disables installation/startup and is the default when using `--mcp-config`.

Failed runs retain their workspace (including the pre-export `edited.blend`
checkpoint); the error prints its path. Successful runs remove temporary files.

The returned USD or manifest is the new Isaac input; source and CT data are
preserved. For Claude, install the `claude` extra and use `--backend claude`.

See [Blender asset editing](blender.md) for supported edits and the output contract.
For geometry edits to navigation bundles, pass `--vessel-names aorta ...` with
the original vessel selection. Use `--help` for session limits and MCP configuration.
