# Working with patient USD files

This guide describes the files authored by
[`patient_digital_twin.exporters`](../patient_digital_twin/exporters/). Commands
run from `patient-digital-twin/`; Python examples use an environment with the
patient package, NumPy and the `usd` extra installed.

## Choose the output

| Output | Produced by | Coordinates and contents |
| --- | --- | --- |
| Standalone `.usd`, `.usda`, `.usdc` | `body.export_to_usd(path, pose="scan")` | Presentation stage; anatomy, optional SOMA, stored centerlines and embedded attached CT |
| `patient_anatomy.usdc` plus `patient_twin.yaml` | `body.export_patient_twin(new_directory)` | Patient-frame anatomy and stored centerlines; CT/attenuation are separate bundle files |
| Both of the above | `examples/pipeline.py` | Adds `human_body.usdc` alongside the patient bundle |
| Physics-demo assets and extended manifest | `export_physics_examples(...)` | Derived meshes/configuration for the supported external physics demos |

All patient stages have default prim `/HumanBody`, `metersPerUnit=1.0`, and
Z-up metadata. `.usda` is useful for small readable examples; `.usdc` is useful
for binary assets. Embedded CT arrays can be large: inspect shapes and attribute
names before printing values or converting a full stage to text.

A patient USD is a static geometry snapshot. It has no animated skeleton,
collision setup, rigid-body dynamics or tissue solver. CT and centerline custom
attributes are data for consumers, not volume or curve rendering primitives.
Exporting a stage does not validate anatomical accuracy or anonymize source data.

## Export and pose

```python
body.export_to_usd("human_body.usdc")
body.export_to_usd("current_pose.usdc", pose="current", skin_opacity=0.15)
manifest = body.export_patient_twin("new_patient_bundle")
```

`body` is a `HumanBody` constructed from imported anatomy. Neither CT nor SOMA
is required. Attach CT with `AttachImaging()` and SOMA with `AttachExternalBody()`
when those are needed; see the [package usage guide](../README.md).

With SOMA, standalone `pose="scan"` evaluates the arms-down presentation and
places the body supine, with head along +X and anterior along +Z. `pose="current"`
uses the displayed pose and the upright axis conversion. Without SOMA, the root
is identity and the structure transforms are used directly. Both exports
preserve the live body's pose and visibility. Do not apply a second axis
conversion to an exported stage.

Bundle anatomy is authored in the manifest's `coordinate_frame`: `DICOM_LPS`
when a source registration exists, otherwise `body`. The manifest's
`transforms.world_from_patient_m` is a separate downstream placement, not baked
into `patient_anatomy.usdc`. Apply it once when placing that bundle into the
consumer's world. It is a NumPy-style column-vector matrix in meters.
`transforms.voxel_to_patient_mm`, when present, instead uses millimeters.

Bundle `soma_pose="scan"` uses the arms-down presentation;
`soma_pose="imaging"` uses the original registration pose. Neither poses the
CT. `exterior="auto"` uses attached SOMA or no exterior. A CT envelope requires
explicit `exterior="ct"` and attached CT. Standalone exports do not generate a
CT envelope.

A standalone USD replaces an existing file only after constructing the new
layer. Bundle and physics exporters require new output directories and publish
their completed output together. Keep all bundle artifacts with their manifest.

## Hierarchy and visibility

```text
/HumanBody                         default Xform
  /Anatomy/<identifier>            one Mesh per retained anatomical structure
  /Exterior/SOMA                  optional attached skin
  /Exterior/CT                    alternative, explicitly requested bundle envelope
  /Looks/<identifier>/Surface     bound UsdPreviewSurface materials
  /Imaging/CT                     optional custom data, standalone export only
```

The exterior and imaging branches are optional. Prim identifiers are sanitized
and may get numeric suffixes; discover anatomy through custom data
`anatomy:name`, not by assuming an exact path from a source label. Other custom
data includes `anatomy:kind` and, where authored, `anatomy:anchor_joint`.
Bundle manifests also map canonical names to `anatomy.structures[name].prim_path`.

Disabled anatomy retains its mesh with `visibility="invisible"`. Absent anatomy
has no mesh prim; the bundle lists it under `anatomy.missing_meshes`. Use computed
visibility to account for hidden ancestors. Hide `/HumanBody/Exterior` to inspect
organs without deleting the skin. Display color and opacity are authored along
with bound materials; changing only a display primvar may not override a material.

## Inspect geometry in Python

This prints names, counts, visibility and one world-space point without dumping
large arrays. Set `path` to either patient USD output.

```python
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom

path = Path("human_body.usdc")
stage = Usd.Stage.Open(str(path.resolve()))
if stage is None:
    raise ValueError(f"Cannot open {path}")
assert stage.GetDefaultPrim().GetPath().pathString == "/HumanBody"
assert UsdGeom.GetStageUpAxis(stage) == "Z"
unit = UsdGeom.GetStageMetersPerUnit(stage)
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
for prim in stage.Traverse():
    name = prim.GetCustomDataByKey("anatomy:name")
    if name is None or not prim.IsA(UsdGeom.Mesh):
        continue
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    # Gf matrices use row-vector convention; transpose for NumPy column math.
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
    world_m = (points @ matrix[:3, :3].T + matrix[:3, 3]) * unit
    print(name, str(prim.GetPath()), len(points),
          UsdGeom.Imageable(prim).ComputeVisibility(), world_m[0])
```

`XformCache` includes the root and all ancestor transforms. Do not apply the
root again. For a bundle, these stage coordinates are still patient coordinates;
apply the manifest's world placement separately when the consumer needs it.

To preview visibility changes without changing the asset on disk:

```python
stage.SetEditTarget(stage.GetSessionLayer())
exterior = stage.GetPrimAtPath("/HumanBody/Exterior")
if exterior:
    UsdGeom.Imageable(exterior).MakeInvisible()
```

Use a separate output layer/file for persistent derived edits. If editing prim
paths or geometry in a bundle, update manifest references and any affected
centerlines too; an independently edited USD does not update the source Python
body or bundle metadata.

## Read embedded CT and centerlines

`/HumanBody/Imaging/CT` has these custom attributes in a standalone export:

| Attribute | Meaning |
| --- | --- |
| `ct:hu` | Flattened float CT samples in HU |
| `ct:shapeZYX` | Three-dimensional array shape |
| `ct:arrayOrder` | `ZYX_C`: C-order flattening of a ZYX array |
| `ct:units` | `HU` |
| `ct:voxelToHuman` | Gf matrix mapping XYZ voxel indices to the `/HumanBody` local frame |

Continue with `stage` from the inspection example:

```python
ct = stage.GetPrimAtPath("/HumanBody/Imaging/CT")
if ct:
    shape = tuple(ct.GetAttribute("ct:shapeZYX").Get())
    hu_zyx = np.asarray(ct.GetAttribute("ct:hu").Get()).reshape(shape, order="C")
    voxel_to_human = np.asarray(ct.GetAttribute("ct:voxelToHuman").Get()).T
    root_to_world = np.asarray(cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath("/HumanBody"))).T
    voxel_to_stage = root_to_world @ voxel_to_human
    # XYZ index (x, y, z) addresses hu_zyx[z, y, x].
    first_voxel_m = (voxel_to_stage @ [0, 0, 0, 1])[:3] * unit
```

The attached input affine is voxel XYZ to RAS meters, with a separate rigid
body-to-RAS registration. Export handles CT orientation conversion; do not infer
spacing from the array shape. Only CT in HU is currently supported by imaging
exporters. Bundle export additionally requires CT aligned to patient axes.
A current posed skin does not imply the CT has been deformed to follow it.

A structure with an extracted centerline has `centerline:points` (local XYZ
meters), `centerline:edges` (index pairs), `centerline:radii` (meters), and
`centerline:coordinateFrame="structure_local_m"`. Apply the mesh's complete
local-to-world transform to its centerline points exactly as for its vertices.
Rigid pose transforms leave radii unchanged. These attributes are not `BasisCurves`.

Bundle `centerlines` entries point to NPZ files containing `points`, `edges`, and
`radii`, in local meters, plus a `local_to_patient` matrix in the YAML. These are
separate from optional composite navigation arrays `centerline_points_mm.npy`
and `centerline_radii_mm.npy`, which use patient-frame millimeters. Navigation
arrays exist only when attached CT and `vessel_names` requested that export.

## Open in Isaac Sim

Use the checked-in viewer with Isaac Sim's Python interpreter:

```bash
/path/to/isaac-sim/python.sh examples/isaac_sim.py view /tmp/patient/human_body.usdc
```

The script starts `SimulationApp` before importing Kit modules, opens the stage,
waits for loading, then frames `/HumanBody` and adds session-only lighting. It
renders until closed and does not save the lighting or camera to the asset.
`usd-core` alone does not provide this application runtime.

For generation from your own inputs, use the [pipeline commands](../examples/README.md).
The separate `isaac_sim.py export` command uses the bundled sample and always
attaches SOMA; it requires that sample's data and SOMA assets.

## Reimporting mesh files

`SimpleImporter` can read triangulated USD files, flatten authored transforms,
convert stage units to meters, and convert Y-up to the package's reference axis.
However, it merges every mesh in each supplied USD into one named structure,
including hidden meshes. It does not reconstruct the patient hierarchy, SOMA,
CT, centerlines, or visibility policy. A whole patient USD is therefore not a
round-trip patient serialization format for `SimpleImporter`.

For already-positioned per-structure meshes, supply explicit `mesh_to_body`
matrices (identity when already in a shared body frame). The Python importer's
omitted transforms use bundled reference placements, whereas the simple pipeline
supplies identity for omitted transforms. STL/OBJ coordinates must already be
XYZ meters.

## Physics-demo exports

Install the patient's `physics` extra and provide the physics source checkout
containing `physics_simulation/endoluminal/xcath/scenes/` and, for liver,
`physics_simulation/surgical/examples/`. Export from an existing bundle:

```python
from patient_digital_twin.exporters import export_physics_examples

manifest = export_physics_examples(
    "patient_bundle/patient_twin.yaml",
    "new_physics_bundle",
    physics_root="/path/to/physics-checkout",
    demos=("aorta",),
)
```

Supported demos are `aorta`, `airways` (requires `trachea`), and `liver`.
Omitting `demos` requests all three, each requiring an enabled source mesh.
The exporter reads source geometry and demo configuration, keeps the largest
mesh component, simplifies large meshes, repairs surfaces, and derives demo
inputs. Aorta/airway exports cut an entry and use a numerical scene scale of 30;
liver export tetrahedralizes and uses scale 20. The manifest records
`scene_from_patient_m` for each demo. These are demo-specific scales, not a claim
that solver parameters are calibrated to SI tissue properties.

The new manifest references original bundle artifacts instead of copying them.
Keep those source files available. Some liver configuration also references the
physics checkout, so the output is not a portable standalone USD package.
`body.export_patient_twin(..., physics_root=...)` can request all physics demos
during bundle creation; use the separate function to select a subset.

The implementation and behavioral checks live in
[usd.py](../patient_digital_twin/exporters/usd.py),
[patient_twin.py](../patient_digital_twin/exporters/patient_twin.py),
[physics_export.py](../patient_digital_twin/exporters/physics_export.py), and
[the export tests](../tests/test_usd.py).
