# Optional agent editing

Generate a patient USD or bundle normally, then optionally edit it with
Claude/Codex and [Blender MCP](https://github.com/ahujasid/blender-mcp).

From the repository root:

```bash
uv pip install -e './patient-digital-twin[pipeline,codex]'
uvx mcp-for-blender install-addon
```

Open Blender, enable its MCP add-on/server, and authenticate Codex. Then run:

```bash
patient-digital-twin-agent --backend codex \
  --input output/patient_twin.yaml --output edited-patient \
  'Smooth the liver surface and give it a red material. Inspect the result.'
```

Blender must run on the same machine. The command creates a separate editing
scene and writes a new output directory; use its returned USD or manifest in
Isaac. The source and CT data are preserved. For Claude, install the `claude`
extra and use `--backend claude` with its SDK authentication configured.

Supported: existing-mesh geometry, transforms, UVs, and USD Preview Surface
materials/image textures. Add/delete/split/join, rigs and animation are unsupported.
For geometry edits to navigation bundles, pass `--vessel-names aorta ...` with
the original vessel selection to regenerate masks and navigation centerlines.
See `patient-digital-twin-agent --help` for session limits and MCP configuration.
