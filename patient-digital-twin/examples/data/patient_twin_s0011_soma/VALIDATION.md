# Full-body patient twin validation

Generated on 2026-09-21 with `examples/export_patient_twin.py` from the s0011 CT,
using NV-Segment CT_BODY followed by SOMA attachment. The manifest lists the SOMA
exterior and all 109 available internal meshes, with 13 missing catalog meshes
reported explicitly. Geometry is stored in the referenced `patient_anatomy.usdc`.

SOMA is evaluated in the saved attachment pose and transformed back into the
original CT frame. Its automatic landmark-fit residuals are 9.7 mm at the left
shoulder and 35.3/32.5 mm at the hips; this is an approximate model registration.
Head/limb geometry outside the acquired CT is visualization only: the measured
CT attenuation remains the source of fluoroscopy.

Compared with the earlier `../patient_twin_s0011` export:

- Both manifest coordinate transforms are identical.
- HU, attenuation, vessel mask, centerline points, edges, and radii are byte-identical.
- Maximum change in the placement of any of the 109 internal meshes: **0 meters**.
- Sampled fluoroscopy frames from both workflows are pixel-identical to their
  corresponding earlier CT-only runs.

The workflow's table adapts to the SOMA exterior instead of moving the patient.
For this subject the tabletop is 1.785 × 0.683 m, centered at
(-0.2693, -0.0122, 0.6321) m. Its upper surface is 2 mm below the lowest skin
vertex, and it extends 5 cm beyond the patient in X and Y. The pedestal remains
on the floor. CT-only bundles keep the previous table.

Visible Isaac Sim rollouts in `/home/mallan/dev/i4h-workflows`:

| Mode | Successes / attempts | Node / workflow completion step | Recorded frames | Exit |
| --- | --- | --- | --- | --- |
| demo | 1/1, one attempt | 118 / 119 | 118 | 0 |
| validate_fluoroscopy | 1/1, one attempt | 72 / 73 | 72 | 0 |

Both use the existing local controllers, without a learned-policy checkpoint.
The scene step cap was unchanged. Recordings passed `i4h-dataset inspect --segments`:

- `/home/mallan/dev/i4h-workflows/runs/endoluminal_navigation/20260921_114305/verify.hdf5`
- `/home/mallan/dev/i4h-workflows/runs/endoluminal_navigation/20260921_114439/verify.hdf5`

Each has one successful segment, three action/state channels, finite states,
and 1024×1024 fluoroscopy. The C-arm check changed orbit by 0.54 radians.
The live scene was inspected for full-body placement and internal anatomy.
The persisted scene was reopened in run `20260921_114557`; actual USD world
bounds confirmed 109 internal meshes and 0.002000011 m skin/table clearance.
See [scene.png](scene.png) and [fluoroscopy.png](fluoroscopy.png).

Patient/example regression tests: **161 passed, 2 skipped**. The independent
workflow table-placement test passed. Workflow `lint --all` passed. Existing
`test_patient_twin.py` could not run in plain pytest because importing its catheter
fixture bootstraps Isaac Sim; coordinate compatibility was checked in the actual
simulator and by comparing all exported artifacts above.

To reopen from the i4h-workflows checkout:

```bash
./run.sh endoluminal_navigation --live \
  --patient-twin /home/mallan/dev/i4h-digital-twin/patient-digital-twin/examples/data/patient_twin_s0011_soma/patient_twin.yaml
```

Use `--mode demo --episodes 1 --attempts 1 --record verify.hdf5`, or
`--mode validate_fluoroscopy --episodes 1 --attempts 1 --record verify.hdf5`,
in place of `--live` to repeat the recorded checks.
