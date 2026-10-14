---
name: patient-digital-twin
description: Import anatomy (segmentations, NV-Segment, NV-Generate, STL/OBJ), attach CT, control visibility, extract vessel centerlines, and export USD or i4h-workflows patient bundles with this repository's patient_digital_twin package. Use for patient package workflows, its CLI, and API changes; use patient-usd for inspecting or consuming exported USD assets.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.9.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - digital-twin
    - patient
    - medical-imaging
    - usd
---

# Patient Digital Twin

`patient_digital_twin` (distribution `patient-digital-twin`, in `patient-digital-twin/`)
turns medical imaging into named anatomy meshes, optional vessel centerlines, and
USD or bundle exports. Work from the repository root unless a command changes
directory; resolve links relative to this file.

- [Quickstart and options](../../patient-digital-twin/README.md): install, the s0011 walkthrough, DICOM, backends.
- [Architecture](../../patient-digital-twin/docs/architecture.md): modules, key classes, data flow, coordinate frames. Read before changing the API.
- [Example scripts](../../patient-digital-twin/examples/README.md) and [USD layout](../../patient-digital-twin/docs/usd.md).

## Purpose

Build a patient digital twin from imaging or meshes and export it for viewing or
for i4h-workflows, using only the documented `patient_digital_twin` API and CLI.

## Instructions

1. Identify the input (label volume, mask folder, CT/MR for NV-Segment, NV-Generate, or STL/OBJ) and the requested output (arrays, USD, or bundle).
2. Install only the extras that path needs (below).
3. Import the named anatomy, wrap it in `HumanBody`, and attach matching CT when the output needs it.
4. Extract centerlines or request bundle `vessel_names` only for vessels the user asked for.
5. Export to a new output path, then verify names, empty meshes, CT grid, and manifest paths.

## Install only what the task needs

```bash
uv pip install './patient-digital-twin[pipeline]'   # USD export + VTK/SciPy centerlines
```

Extras: `pipeline` (usd-core, vtk, scipy), `usd`, `dicom` (SimpleITK), `nvsegment`,
`nvgenerate` (inference dependencies only; model checkouts and weights are separate),
and `dev` (pytest). STL/OBJ import needs `trimesh`. The root aggregate package uses
the same extras prefixed with `patient-`. `usd-core` does not install Isaac Sim.

## Task → call

| Task | Call |
| --- | --- |
| Mesh a labeled NIfTI | `SegmentationImporter("labels.nii.gz", labelmap).to_anatomy_collection()` (labelmap: ID->name dict, or NV-Generate name->ID JSON path) |
| Mesh a folder of binary masks | `SegmentationImporter("masks/", names=["aorta"]).to_anatomy_collection()` (filenames are names) |
| Mesh a NumPy ZYX label array | `SegmentationImporter(labels_zyx, {1: "liver"}, affine_xyz_to_imaging_m=affine)` |
| Segment a CT/MR with NV-Segment | `NVSegmentImporter("ct.nii.gz", bundle_root=root).to_anatomy_collection(names=["aorta"])` |
| Generate a synthetic patient | `gen = NVGenerateImporter(source_root=root)`; `gen.to_anatomy_collection(names=[...])`; CT is `gen.ct_scan` |
| Import STL/OBJ meshes | `SimpleImporter({"liver": "liver.stl"}, mesh_to_body={...}).to_anatomy_collection()` |
| Wrap anatomy | `body = HumanBody(anatomy)` |
| Attach CT | `body.attach_scan(scan_volume.from_nifti("ct.nii.gz"))` (or `from_dicom`, `load_artifact`) |
| Hide / show | `structure.enabled = False`, `body.anatomy.set_structure_enabled / set_system_enabled / set_enabled` |
| Query | `body.anatomy.structures[name]`, `body.anatomy.select(kind=Kind.VESSEL, include_empty=False)` |
| Vessel centerlines | `body.extract_topology(names=["aorta"], spacing_m=0.0015)` |
| Standalone USD | `body.export_to_usd("patient.usdc")` |
| i4h-workflows bundle | `body.export_patient_twin("bundle", vessel_names=["aorta"], ct_exterior=True)` |
| CT + vessel arrays only | `export.write_artifacts(scan, "out", vessel_mask=mask)` |
| End-to-end inference CLI | `python -m patient_digital_twin --source nvsegment --input ct.nii.gz --bundle-root ROOT --classes aorta --format bundle --output out` |

All public classes import from `patient_digital_twin`; `scan_volume`, `export`, and
`geometry` are imported as submodules. `run_pipeline(**kwargs)` in
`patient_digital_twin.__main__` is the CLI as a Python call.

## Rules that are easy to get wrong

- Label IDs are backend-specific: always pass the label dictionary that matches the
  segmentation. Names are normalized with `canonical_name` ("Left Kidney" → `kidney_left`).
- Geometry is XYZ meters; grids/masks are ZYX. `structure.vertices` are local;
  `body_vertices` use `local_to_body` (import placement); `world_vertices` use
  `local_to_world` (current pose). Do not transform `world_vertices` again.
- Disabled structures keep their mesh (`structure.mesh`) but public geometry returns
  `None`; exports keep them as invisible prims. To omit structures, select names at import.
- `attach_scan` keeps the scan's native array order, frame (RAS/LPS), and units; nothing
  resamples. CT must be on the segmentation's physical grid for bundle vessel masks.
- `extract_topology` needs closed vessel/airway meshes and VTK; it stores graphs only if
  every extraction succeeds. Bundle navigation arrays (`centerline_*.npy`) are computed
  separately on the CT grid from `vessel_names`.
- Output directories (and the CLI's output) must be new. `export_to_usd` replaces a file.
- In-process NV inference changes the working directory under a lock; pass
  `python_executable=` (CLI `--python`) to run the model in another environment.
- `scan_volume.py` is kept byte-identical with sensor-simulation; do not edit it here.
- `vasculature_digital_twin` (same distribution) is deprecated: do not use it for new work.
  Map its calls to `patient_digital_twin` with the README's "Deprecated" section.

## Prerequisites

Python 3.10+ and the package extras for the chosen path. The s0011 CT is a Git LFS
file (`git lfs pull`); the s0011 masks need no model inference. NV-Segment and
NV-Generate need an upstream checkout, model weights, and CUDA-compatible PyTorch.
Isaac Sim is needed only for interactive viewing.

## Limitations

Exports are static geometry, optional centerlines, and native-HU CT. They are not a
tissue simulation, a deformable model, or an anonymization step. Sheared CT grids
need explicit resampling before bundle centerlines. HU-to-attenuation mapping and
simulator placement belong to the consuming workflow.

## Verify

Check expected structure names, present versus empty meshes (`structure.is_empty`), the
registration, and output manifest paths; inspect USD with the patient-usd skill. After
code changes, from `patient-digital-twin/` run:

```bash
uv run --extra dev pytest
uv run --extra dev mypy
```

Report optional-dependency skips separately from tested behavior. Do not describe a
geometry export as a validated tissue simulation or an anonymization step. If inputs,
dependencies, or registration are missing, report what is missing instead of switching
patients or backends.

## Troubleshooting

- Wrong or missing structure names: check that the label dictionary matches the backend that produced the labels, and compare against `canonical_name` output.
- Export fails because the output exists: choose a new directory (bundles, CLI output); `export_to_usd` replaces only a single file.
- Bundle vessel mask or centerline errors: confirm the CT shares the segmentation grid and that the requested `vessel_names` were imported.
- `extract_topology` stores nothing: install `vtk`/`scipy` (`pipeline` extra) and check the meshes are closed.
- NV model import errors: rerun with `--python` pointing at the environment that has the model dependencies.

## Examples

- `Build a navigation bundle for the s0011 aorta with its CT.` → import the s0011 aorta mask, attach `ct.nii.gz`, call `export_patient_twin(..., vessel_names=["aorta"], ct_exterior=True)` into a new directory, and report the manifest artifacts.
- `Segment the aorta from my CT with NV-Segment and export a bundle.` → run `python -m patient_digital_twin --source nvsegment --input ct.nii.gz --bundle-root ROOT --classes aorta --format bundle --output NEW_DIR`; report missing weights instead of switching backends.
- `Hide the liver but keep it in the USD.` → set `enabled = False` on the structure and export; the liver remains as an invisible prim.

## Completion gate

Report the inputs, extras, structures imported (present vs empty), CT attachment and
frame, output paths, verification performed, and any optional-dependency skips.
