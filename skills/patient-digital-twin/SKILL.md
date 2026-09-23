---
name: patient-digital-twin
description: Import anatomy, attach imaging or SOMA, configure and pose HumanBody, extract topology, and export patient bundles with this repository's patient_digital_twin package. Use for patient package workflows and API migration; use patient-usd for inspecting or consuming exported USD assets.
---

# Patient Digital Twin

Use the repository's independently installable `patient-digital-twin` distribution
and `patient_digital_twin` import namespace. Work from the repository root unless
a command explicitly changes directory. Shared skills live in `skills/`;
`.codex/skills` and `.claude/skills` point there.

Read [the usage guide](../../patient-digital-twin/README.md) for installation,
coordinates and SOMA options. Read [the pipeline commands](../../patient-digital-twin/examples/README.md)
when generating an end-to-end output, and [the architecture guide](../../patient-digital-twin/patient_digital_twin/README.md)
when changing the API. Resolve these paths relative to this skill's directory.

## Select an input and runtime

- Existing label image: `SegmentationImporter(path, matching_label_dictionary)`.
  Numeric label IDs are workflow-specific; do not substitute another backend's map.
- Existing named meshes: `SimpleImporter` from `patient_digital_twin.importers`.
  STL/OBJ must use XYZ meters; USD imports bake units and transforms. Supply
  explicit `mesh_to_body` matrices for your data. Omitted Python API placements
  use the bundled reference patient, not identity.
- CT requiring segmentation: `NVSegmentImporter` or `TotalSegmentatorImporter`.
  Fresh paired generation: `NVGenerateImporter`. Read the relevant adapter's
  constructor before choosing backend options; models/runtimes are separate
  dependencies. NVIDIA adapters can use a separate Python interpreter.

Install only integrations required for the task: from the root,
`uv pip install './patient-digital-twin[usd]'` enables USD. Patient extras are
`soma`, `viewer`, `usd`, `physics`, and `dev`; root extras prefix the integration
names with `patient-`. Skeleton topology also needs `vtk` and `scipy`.
SOMA needs real model assets, not Git LFS pointer files. `usd-core` does not
install Isaac Sim. Use the caller's provided input paths; sample data is not a
substitute for the requested patient.

## Build and use the body

```python
from patient_digital_twin import HumanBody, SegmentationImporter

anatomy = SegmentationImporter("segmentation.nii.gz", "labels.json").to_anatomy_collection()
body = HumanBody(anatomy)
body.anatomy.set_structure_enabled("liver", True)
body.export_to_usd("patient.usdc")
```

Importers return `AnatomyCollection`; they do not attach SOMA. `body.soma` and
`body.imaging` start as `None`. Use `body.anatomy` for configuration, systems and
selection. Disabled geometry remains in `structure.mesh`, while public geometry
properties return `None`; export keeps it invisible. To omit structures entirely,
use the pipeline's `--anatomy` selection. Keep bones available until SOMA matching
is complete.

All geometry and rigid transforms use XYZ meters. Image arrays use ZYX indexing;
voxel affines take XYZ indices. `body.anatomy.body_to_imaging` preserves import
registration; `local_to_body` is the original mesh placement and `local_to_world`
is the current pose. Do not transform `world_vertices` a second time.

Attach matching CT using `body.AttachImaging(volume_zyx,
voxel_to_imaging=voxel_to_ras_m)`, with an explicit `body_to_imaging` when the
image uses another frame. CT must already be HU. `source_path` is provenance,
not a file loader. Exporters do not accept the old `ct_path` argument.

For articulation, call `body.AttachExternalBody(...)`, then use `body.soma.pose`,
`fit_soma_shape`, `fit_bone_anchors`, and `check_containment`. Automatic attachment
requires at least three non-collinear shoulder/hip matches; partial scans may
need measured body-frame landmarks or explicit `body_to_soma` registration.
Do not fabricate registration or resize internal anatomy to hide a failed fit.
Save/reload attachment options through `body.soma.soma_configuration`; this does
not save the anatomy meshes. Assess containment for the patient's required poses.

## Topology and export

`body.extract_topology(spacing_m=0.0015)` selects skeleton extraction with a grid
per structure; closed tubular meshes and VTK/SciPy/scikit-image are needed.
Calling without spacing or a method uses VMTK. Extraction processes retained
vessels/airways, including disabled ones, and updates results only after success.
Stored centerlines use each mesh's local meters.

For a pipeline, run from `patient-digital-twin/`:

```bash
uv run --extra usd --with scipy --with vtk python examples/pipeline.py \
  --source segmentation --input /path/to/labels.nii.gz \
  --labels /path/to/labels.json --ct /path/to/matching_ct.nii.gz \
  --output /tmp/new_patient
```

This writes both a bundle and `human_body.usdc`. Add `--soma` only when requested
and installed; `--parameters` requires `--soma`. Simple pipeline input skips CT
and SOMA. The segmentation pipeline requires matching CT; use the Python API for
segmentation-only geometry exports. Output directories must be new.

`export_to_usd` defaults to scan presentation; request `pose="current"` for a
posed snapshot. `export_patient_twin` allows anatomy-only bundles. Navigation
artifacts require attached CT and explicit `vessel_names`; `exterior="auto"`
does not create a CT envelope. Read [the USD guide](../../patient-digital-twin/docs/usd.md)
for coordinate frames, optional artifacts and physics-demo exports.

## Verify the requested result

Check expected structure names, retained versus missing geometry, registration
and output manifest paths. Inspect USD using the patient-usd skill when needed.
Run `uv run --extra dev pytest` from the root or patient distribution after code
changes; report optional integration skips separately from tested behavior.
Do not describe a geometry export as a validated tissue simulation or an
anonymization step. Preserve the user's requested scope when dependencies or
registration are missing; report the concrete missing input instead of silently
switching patients or backends.
