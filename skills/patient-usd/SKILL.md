---
name: patient-usd
description: Inspect, display, transform, or consume USD and patient_twin.yaml assets exported by patient_digital_twin, including embedded CT, centerlines, Isaac Sim viewing. Use for this repository's patient USD contract, not general USD authoring.
---

# Patient USD assets

Read [Working with patient USD files](../../patient-digital-twin/docs/usd.md) for
runnable inspection examples and field definitions. Use
[the patient workflow skill](../patient-digital-twin/SKILL.md) when the task needs
to create or change the source `HumanBody`. Resolve links relative to this file.

## Identify the asset before using it

Inspect file paths and, when present, `patient_twin.yaml` before assuming a
coordinate frame or available imaging:

| Asset | Interpretation |
| --- | --- |
| Standalone `human_body.usdc` or another `export_to_usd` file | Presentation stage; CT may be embedded |
| Manifest's `artifacts.anatomy_usd` (normally `patient_anatomy.usdc`) | Patient-frame anatomy; CT and vessel arrays are separate manifest artifacts |

Names alone are not proof: inspect the stage and manifest. Patient stages have
`/HumanBody` as default prim, declared `metersPerUnit`, and Z-up metadata. Exterior, CT and
centerlines are optional. Summarize hierarchy, names, visibility and array shapes
before loading or printing large voxel arrays. Open stages using `pxr.Usd`, not
text parsing of binary `.usdc` files.

## Preserve geometry and coordinate meaning

- Discover anatomy via `anatomy:name` custom data or manifest `prim_path` entries.
  Sanitized identifiers may differ from source names. `anatomy:kind` identifies
  structure kind.
- Use `UsdGeom.XformCache` for complete local-to-world transforms, including
  `/HumanBody`. Transpose Gf matrices when using the package's NumPy column-vector
  convention. Respect `metersPerUnit`; do not apply a second axis conversion.
- Standalone export preserves current placements, transformed into the scan frame.
- Schema-3 bundle anatomy uses the scan physical frame (`RAS`/`LPS`) and units.
  Apply simulator placement downstream, after converting declared scan units.
- Use computed visibility, including ancestors. Disabled meshes are retained
  but invisible; missing meshes have no prim. Hide `/HumanBody/Exterior` for
  inspection instead of removing skin. Preview changes in a session layer;
  persist requested edits to a deliberate output and maintain bundle references.

The exporter authors static meshes and materials. It does not author animated
rigs, collisions, dynamics, or a CT volume renderer. Changing the Python body
after export does not update the saved stage.

## CT and topology

Standalone CT stores `ct:hu`, `ct:shape`, `ct:arrayOrder`, and
`ct:arrayIndexToScan`, plus physical frame and units. Reshape in C order using the
recorded shape; the full affine maps array indices to scan coordinates.

USD mesh centerlines use local stage units. Bundle graph arrays use scan units and
physical coordinates. Read `patient_twin.yaml` and `volume.yaml`; do not assume
ZYX, LPS, or millimeters. `geometry.py` calculates graphs and `export.py` writes them.

## View or reuse

For interactive viewing, run from `patient-digital-twin/` using the installed
Isaac Sim runtime:

```bash
/path/to/isaac-sim/python.sh examples/isaac_sim.py view /path/to/human_body.usdc
```

The checked-in script starts `SimulationApp` before Kit imports and adds
session-only light/camera changes. Plain `usd-core` suffices for file inspection,
not this viewer. Do not install a simulator just to inspect metadata.

`SimpleImporter` reads STL/OBJ only, one file per named structure; a patient USD
is not a round-trip format. Supply explicit `mesh_to_body` placements.

## Check the result

After an edit or export, reopen the output. Check default prim, units, up axis,
expected named meshes, computed visibility and transforms. If relevant, check CT
shape/affine and centerline alignment without dumping full data. For bundle edits,
resolve affected artifact and prim paths and verify coordinate units. Test the
requested runtime behavior when available and distinguish file validation from
an Isaac Sim session .
