---
name: patient-usd
description: Inspect, display, transform, or consume USD and patient_twin.yaml assets exported by patient_digital_twin, including embedded CT, centerlines, Isaac Sim viewing, and physics-demo bundles. Use for this repository's patient USD contract, not general USD authoring.
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
| Manifest's `artifacts.anatomy_usd` (normally `patient_anatomy.usdc`) | Patient-frame anatomy; CT/attenuation are separate manifest artifacts |
| Physics-demo output | Derived geometry with recorded scene transforms and external source/configuration dependencies |

Names alone are not proof: inspect the stage and manifest. Patient stages have
`/HumanBody` as default prim, meters and Z-up metadata. Exterior, CT and
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
- Standalone export defaults to `pose="scan"`, a supine arms-down presentation
  when SOMA is attached. `pose="current"` selects the displayed pose. Without
  SOMA, the root is identity and current structure placements are retained.
- Bundle anatomy is in `coordinate_frame` (`DICOM_LPS` or `body`). Apply manifest
  `world_from_patient_m` once for consumer world placement; it is not baked into
  the bundle USD. Do not apply that matrix to the standalone presentation file.
- Use computed visibility, including ancestors. Disabled meshes are retained
  but invisible; missing meshes have no prim. Hide `/HumanBody/Exterior` for
  inspection instead of removing skin. Preview changes in a session layer;
  persist requested edits to a deliberate output and maintain bundle references.

The exporter authors static meshes and materials. It does not author animated
SOMA rigs, collisions, dynamics, or a CT volume renderer. Posing the Python body
after export does not update the saved stage.

## CT and topology

Standalone `/HumanBody/Imaging/CT` stores `ct:hu`, `ct:shapeZYX`,
`ct:arrayOrder="ZYX_C"`, `ct:units="HU"`, and `ct:voxelToHuman`.
Reshape HU in C order to ZYX. Transform XYZ voxel indices with `ct:voxelToHuman`
then the HumanBody root transform. Do not assume an identity root, infer voxel
spacing from shape, or treat a posed skin as a deformed CT.

Mesh `centerline:points` and `centerline:radii` use local meters;
`centerline:edges` indexes the points. Apply the mesh's complete transform to
points; rigid transforms leave radii unchanged. The attributes do not render
curves by themselves.

Resolve bundle paths relative to `patient_twin.yaml`. Per-structure centerline
NPZ files use local meters and YAML `local_to_patient`. Optional composite
navigation `centerline_points_mm.npy` and `centerline_radii_mm.npy` use patient
millimeters. Check artifact presence instead of treating an anatomy-only bundle
as a complete navigation input.

## View, reuse, or export physics

For interactive viewing, run from `patient-digital-twin/` using the installed
Isaac Sim runtime:

```bash
/path/to/isaac-sim/python.sh examples/isaac_sim.py view /path/to/human_body.usdc
```

The checked-in script starts `SimulationApp` before Kit imports and adds
session-only light/camera changes. Plain `usd-core` suffices for file inspection,
not this viewer. Do not install a simulator just to inspect metadata.

Do not feed an entire patient USD to `SimpleImporter` expecting a round trip:
it merges all meshes, including hidden ones, into a single named structure per
input file and loses patient-level CT, SOMA, topology and policy state. For
per-structure import, use triangulated assets and explicit `mesh_to_body` placement.

For physics requests, read the guide's physics section before calling
`export_physics_examples`. It takes an existing manifest, a new output directory,
a physics checkout and optional `demos`. Supported inputs are enabled `aorta`,
`trachea` for `airways`, and `liver`; omitting `demos` requests all three.
Derived geometry is repaired/simplified and uses demo-specific scales recorded
in `scene_from_patient_m`. Keep original bundle artifacts and referenced checkout
files available. Do not present these demo scales as calibrated tissue physics.

## Check the result

After an edit or export, reopen the output. Check default prim, units, up axis,
expected named meshes, computed visibility and transforms. If relevant, check CT
shape/affine and centerline alignment without dumping full data. For bundle edits,
resolve affected artifact and prim paths and verify coordinate units. Test the
requested runtime behavior when available and distinguish file validation from
an Isaac Sim session or a physics run.
