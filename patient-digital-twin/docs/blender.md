# Blender asset editing

Install `./patient-digital-twin[pipeline]` and Blender (tested with 5.1.2).
Prepare an existing patient USD or bundle in your Python environment:

```python
from pathlib import Path
from patient_digital_twin.agent import blender
from patient_digital_twin.agent.assets import prepare, finalize

workspace = Path("output/edit").resolve()
prepare(Path("output/patient_twin.yaml"), workspace)
print(f"exec(open({blender.__file__!r}).read()); open_asset({str(workspace)!r})")
print(f"export_asset({str(workspace)!r})")
```

In Blender's Python Console, run the first printed command to open a separate
scene. Edit the meshes/materials, return to Object Mode, then run the second
printed command to export. Blender must share the same machine/filesystem.
Back in your original Python environment, publish the final asset:

```python
result = finalize(workspace, Path("output/edited-patient").resolve())
print(result)  # Use this USD or patient_twin.yaml in Isaac.
```

The output directory must be new. Only manifest-listed bundle files and required
textures are published; source files, CT data and original prim identities are
preserved. Navigation bundles require `vessel_names=["aorta", ...]` when geometry
changes, using the original vessel selection to regenerate masks and centerlines.

Supported: existing meshes, transforms, UVs and USD Preview Surface materials/image
textures. Add/delete/split/join, rigs and animation are unsupported.
