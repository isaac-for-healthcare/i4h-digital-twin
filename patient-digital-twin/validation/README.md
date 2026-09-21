> Historical results: these reports and parameter files predate the rigid-only
> anatomy/body-frame refactor. Regenerate parameters and containment reports
> with the current viewer; the old deformation results do not apply.

# Validation artifacts

Validated on 2026-09-17 with the local SOMA-X 0.2.1 implementation, CPU dense
skinning, low-LOD SOMA skin, and NV-Generate-CTMR's 132-class label dictionary.

## Final results

- [Final fitting and containment report](soma_containment_final.json):
  109/109 structures pass in scan, arms-out, seated and torso-bend poses.
- [Independent configuration reload](reload_containment.json):
  the viewer CLI reproduced all four passes with exit status 0.
- [Reloadable SOMA configuration](fitted_soma_configuration.json):
  identity, bone scales, scan pose, imaging registration and rigid anchor frames.
- Screenshots: [scan](viewer_scan.png), [arms out](viewer_arms_out.png),
  [seated](viewer_seated.png), [torso bend](viewer_torso_bend.png).
  Chrome exercised the actual pose controls and reported no JavaScript errors.
  The screenshots use software WebGL in the headless test browser.

Each pose checks all 3,416,804 mesh vertices and triangle centers, not a
random subset. This is a finite-point containment check, not a mathematical
proof against every possible triangle intersection or arbitrary future pose.
Bone vertices and topology are unchanged. Soft tissues use anatomically
filtered SOMA skinning weights. Contact correction was enabled, but no
projection was needed in the final four-pose run.

The source volume is:
`/home/mallan/dev/NV-Generate-CTMR/temp_work_dir/datasets/all_masks_flexible_size_and_spacing_4000/TotalSegmentatorV2/s0011/ct_label_wbdm.nii.gz`.
It is an existing source mask distributed for the NV-Generate workflow.
The four generated output volumes available locally lacked usable shoulder/
hip landmarks, so they could be imported but could not be automatically
registered by the reference landmark method.

The initial rigid-only report and intermediate fitting artifacts are retained
for comparison. Use files named **final**, **reload**, and
**fitted_soma_configuration** for the completed result.

## Reopen the checked viewer

From the i4h-digital-twin repository root, using its own virtual environment:

```bash
uv sync --extra dev --extra patient-soma --extra patient-viewer
.venv/bin/python patient-digital-twin/examples/viewer.py \
  --segmentation /home/mallan/dev/NV-Generate-CTMR/temp_work_dir/datasets/all_masks_flexible_size_and_spacing_4000/TotalSegmentatorV2/s0011/ct_label_wbdm.nii.gz \
  --labels /home/mallan/dev/NV-Generate-CTMR/configs/label_dict.json \
  --parameters patient-digital-twin/validation/fitted_soma_configuration.json \
  --deform-soft-tissues --port 8091
```

Add `--validate-only --report patient-digital-twin/validation/reload_containment.json` for
noninteractive verification. General installation and fitting instructions
are in the [package README](../README.md).

## Code checks

The full optional-dependency suite passes 142 tests, including actual SOMA
evaluation, soft-tissue deformation, rigid-bone fitting, configuration reload,
YAML mesh policies, and live viewer HTTP/scene tests. Configuration tests cover
restoration at the current pose, including direct SOMA calls while meshes are
disabled. Direct tests now cover every public package class, including a
shape-responsive fitting fixture and a controlled TotalSegmentator mapping
loader. See the [class-to-test map](../patient_digital_twin/README.md#source-map-and-unit-tests).
Ruff and whitespace checks pass.

The standalone minimal patient package passes 112 tests with 4 optional-integration
skips. The complete 142-test run uses the repository-root `.venv`, PyPI
`py-soma-x==0.2.1`, and `PATIENT_TWIN_TEST_SOMA=1`. Meshing is included in
the base package, and isolated-import tests require no source-path overrides.
Both standalone and aggregate wheels also pass clean-install meshing checks.
