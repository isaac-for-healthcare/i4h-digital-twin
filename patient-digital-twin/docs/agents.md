# Optional agent editing

Generate a patient USD or bundle normally. To modify its geometry, materials or
textures, run a separate editing step using Claude/Codex and upstream
[Blender MCP](https://github.com/ahujasid/blender-mcp). Without that step, the
existing pipeline is unchanged.

```bash
uv pip install -e './patient-digital-twin[pipeline,codex]'
patient-digital-twin-agent --backend codex \
  --input output/patient_twin.yaml --output edited-patient \
  'Smooth the liver surface and apply a red textured material. Inspect the result.'
```

Use the `claude` extra and `--backend claude` for Claude. Configure provider
credentials through the corresponding SDK. Blender must already be running with
its upstream MCP add-on enabled, on the same machine/filesystem. The default
server command is `uvx mcp-for-blender`; `--mcp-config config.json` accepts a stdio
server configuration (`command`, `args`, optional `env`) for a pinned/local install.
This package does not install or manage Blender. The handoff was tested on Blender
5.1.2; other versions must expose the USD import/export options used by the helper.

The agent edits a separate Blender scene. The application imports the asset,
lets the agent inspect/edit it, exports the result, then publishes a new output
directory. The returned USD or `patient_twin.yaml` is the final downstream input.
The source remains intact; the Blender scene remains open for inspection.

## Supported edits and output contract

- Existing named meshes: deformation, modifiers, transforms, UVs and materials.
  Keep mesh-data `i4h_path` identities. Adding, deleting, splitting or joining
  meshes is rejected. Leave Edit Mode before finishing.
- Use USD Preview Surface compatible materials, such as Principled BSDF with
  image textures. Bake unsupported procedural shaders before export. Referenced
  texture files are copied into the output; arbitrary Blender shaders, rigs and
  animation are not a supported interchange contract.
- Finalization keeps original prim paths, stage units, placements, anatomy
  metadata and CT attributes. Blender geometry is converted back to each original
  mesh's local coordinates. Original CT data is never sent through Blender.
- Changed meshes lose their stored centerlines. If a bundle has navigation arrays,
  geometry edits require `--vessel-names aorta ...` specifying the original vessel
  selection, and regenerate the mask and navigation centerlines on the original
  scan grid. Failures do not publish an output. CT intensities are not regenerated.
- Bundle metadata and CT files are retained. The modified USD/bundle follows the
  existing Isaac loading contract. File-level validation and a real Blender
  round trip are covered, and the upstream MCP handoff was exercised. An edited
  s0011 bundle was opened and rendered in Isaac Sim 6.2, including the red liver
  material. Arbitrary shader compatibility still depends on the target runtime.

`--timeout`, `--max-turns` and `--budget-usd` bound the SDK session (the latter two
are Claude options). The agent can execute Blender Python: use trusted local
sessions. The source-preservation contract is enforced by application writes;
Blender Python execution itself is not a filesystem sandbox.
