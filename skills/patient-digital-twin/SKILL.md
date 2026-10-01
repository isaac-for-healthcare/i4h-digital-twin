---
name: patient-digital-twin
description: Import anatomy, attach imaging, configure HumanBody, extract topology, and export patient bundles with this repository's patient_digital_twin package. Use for patient package workflows and API migration; use patient-usd for inspecting or consuming exported USD assets.
---

# Patient Digital Twin

Use the repository's independently installable `patient-digital-twin` distribution
and `patient_digital_twin` import namespace. Work from the repository root unless
a command explicitly changes directory. Shared skills live in `skills/`;
`.codex/skills` and `.claude/skills` point there.

Read [the usage guide](../../patient-digital-twin/README.md) for installation,
coordinates and imaging options. Read [the pipeline commands](../../patient-digital-twin/examples/README.md)
when generating an end-to-end output, and [the architecture guide](../../patient-digital-twin/patient_digital_twin/README.md)
when changing the API. Resolve these paths relative to this skill's directory.

## Select an input and runtime

- Existing label image: `SegmentationImporter(path, matching_label_dictionary)`.
  Numeric label IDs are workflow-specific; do not substitute another backend's map.
- Existing named meshes: `SimpleImporter` from `patient_digital_twin.importers`.
  STL/OBJ must use XYZ meters; USD imports bake units and transforms. Supply
  explicit `mesh_to_body` matrices for your data. Omitted Python API placements
  are identity; no sample-based placement is inferred.
- CT requiring segmentation: `NVSegmentImporter`.
  Fresh paired generation: `NVGenerateImporter`. Read the relevant adapter's
  constructor before choosing backend options; models/runtimes are separate
  dependencies. NVIDIA adapters can use a separate Python interpreter.

Install only integrations required for the task: from the root,
`uv pip install './patient-digital-twin[usd]'` enables USD. Patient extras are
`pipeline`, `dicom`, `usd`, and `dev`; root extras prefix the integration
names with `patient-`. Skeleton topology also needs `vtk` and `scipy`.
`usd-core` does not install Isaac Sim. Use the caller's provided input paths;
sample data is not a substitute for the requested patient.

## Build and use the body

```python
from patient_digital_twin import HumanBody, SegmentationImporter

anatomy = SegmentationImporter("segmentation.nii.gz", "labels.json").to_anatomy_collection()
body = HumanBody(anatomy)
body.anatomy.set_structure_enabled("liver", True)
body.export_to_usd("patient.usdc")
```

Importers return `AnatomyCollection`; `body.imaging` starts as `None`.
Use `body.anatomy` for configuration, systems and selection. Disabled geometry remains in `structure.mesh`, while public geometry
properties return `None`; export keeps it invisible. To omit structures entirely,
use the pipeline's `--anatomy` selection.

Internal anatomy geometry and rigid transforms use XYZ meters. `local_to_body`
is the original mesh placement and `local_to_world` is the current placement.
Use `body.attach_scan(scan_volume.from_nifti(...))` for native CT exports; lower-level
`attach_imaging` accepts KJI arrays and an IJK-to-RAS-meter affine. Schema-3 exports
retain source array order, physical frame, and units. Simulator placement belongs
to the consumer; attenuation mapping belongs to sensor-simulation.

## Topology and export

`body.extract_topology(spacing_m=0.0015)` voxelizes each structure on a grid and
skeletonizes it; closed tubular meshes and VTK/SciPy/scikit-image are needed.
Extraction processes retained
vessels/airways, including disabled ones, and updates results only after success.
Stored centerlines use each mesh's local meters.

For a pipeline, run from `patient-digital-twin/`:

```bash
uv run --extra usd --with scipy --with vtk python examples/pipeline.py \
  --source segmentation --input /path/to/labels.nii.gz \
  --labels /path/to/labels.json --ct /path/to/matching_ct.nii.gz \
  --output /tmp/new_patient
```

This writes both a bundle and `human_body.usdc`. Simple pipeline input skips CT.
The segmentation pipeline requires matching CT; use the Python API for segmentation-only geometry exports. Output directories must be new.

`export_to_usd` preserves current structure transforms with an identity root.
`export_patient_twin` allows anatomy-only bundles. Navigation artifacts require attached CT and explicit `vessel_names`; pass `ct_exterior=True`
for a CT envelope. Read [the USD guide](../../patient-digital-twin/docs/usd.md)
for coordinate frames and optional artifacts.

## Verify the requested result

Check expected structure names, retained versus missing geometry, registration
and output manifest paths. Inspect USD using the patient-usd skill when needed.
Run `uv run --extra dev pytest` from the root or patient distribution after code
changes; report optional integration skips separately from tested behavior.
Do not describe a geometry export as a validated tissue simulation or an
anonymization step. Preserve the user's requested scope when dependencies or
registration are missing; report the concrete missing input instead of silently
switching patients or backends.
