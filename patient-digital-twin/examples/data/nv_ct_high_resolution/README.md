# High-resolution NV-Generate-CTMR example

This example pairs a **database-derived segmentation** with a **new synthetic CT**
generated using NV-Generate-CTMR's `rflow-ct` image-from-mask workflow.
The mask is not a newly diffusion-generated anatomy or the same subject as the
previous `TotalSegmentatorV2/s0011` example.

| Property | Value |
| --- | --- |
| Shape (XYZ) | 512 × 512 × 768 |
| Spacing (XYZ) | 0.763 × 0.763 × 0.7875 mm |
| Field of view | approximately 391 × 391 × 605 mm |
| Generation | 30 rectified-flow steps, seed 42, guidance 0 |
| Segmentation | uint8 MAISI label IDs; body = 200, background = 0 |
| CT | int16 HU, rounded from generated values (at most 0.5 HU error) |

The input mask was already at this grid resolution in NVIDIA's distributed
database. Its voxel values and affine are preserved; only storage type and
spatial-unit metadata were normalized. No upsampling or smoothing was applied.
This is roughly twice the sampling resolution along each axis of the previous
1.82 × 1.82 × 1.68 mm example, but a finer grid does not guarantee better label
accuracy or eliminate every stair-step artifact.

## Files

- `segmentation.nii.gz`: complete multi-label mask, including body envelope.
- `ct.nii.gz`: generated CT on the same grid; not required by the mesh viewer.
- `labels.json`: matching label dictionary with explicit background/body entries.
- `viewer_parameters.json`: explicit partial-scan pose with full arm lengths preserved.
- `provenance.json`: source identity, generator revision, model/output hashes and command.
- `inference.json`, `network.json`, `environment.json`, `generation.log`: run record.

The compressed segmentation is about 5.5 MiB; the CT is about 119 MiB. The CT
exceeds GitHub's ordinary single-file limit: use Git LFS or external artifact
storage before publishing it. The viewer needs only the much smaller mask,
label dictionary and viewer parameters. Nothing has been uploaded or committed.

## View

From the `i4h-digital-twin` repository root, with the patient viewer extras installed:

```bash
.venv/bin/python patient-digital-twin/examples/viewer.py \
  --segmentation patient-digital-twin/examples/data/nv_ct_high_resolution/segmentation.nii.gz \
  --labels patient-digital-twin/examples/data/nv_ct_high_resolution/labels.json \
  --parameters patient-digital-twin/examples/data/nv_ct_high_resolution/viewer_parameters.json \
  --anatomy-config patient-digital-twin/examples/data/nv_ct_high_resolution/anatomy.yaml \
  --pose scan --port 8093
```

Open http://127.0.0.1:8093 after reconstruction finishes. The full-resolution
mask takes more time and memory to reconstruct. Use **Visible structures** to
select organs. All imported anatomy moves rigidly. The display starts in the arms-down `scan` pose; scan-based attachment is established first so bones move with their arms.
The example anatomy policy disables the out-of-field skull.

The scan contains only 4–6 cm of the superior humeri and no observed elbows.
The saved pose uses the progression of humeral cross-section centers to estimate
upper-arm direction, while preserving full SOMA arm lengths. Forearms remain
straight because elbow flexion cannot be measured from this scan. The saved
`body_to_soma` keeps the trunk registration fixed. `arm_alignment.json` records
the estimated directions and modeled lengths. This is a partial-scan preview,
not a measured full-body pose.

Automatic limb-length fitting must not treat the scan cuts as elbows. The
importer rejects endpoints touching **any** scan boundary, including oblique
bones whose principal axis differs from the cropping axis. Alignment and skin
containment still require checking; the original shoulder registration has an
approximately 27 mm residual.

## Reproduce

Use an NV-Generate-CTMR checkout with its dependencies, mask database, and
`autoencoder_v1.pt`, `diff_unet_3d_rflow-ct.pt`, `controlnet_3d_rflow-ct.pt` installed:

```bash
/home/mallan/dev/NV-Generate-CTMR/.venv/bin/python \
  patient-digital-twin/examples/generate_high_resolution_data.py \
  --nv-root /home/mallan/dev/NV-Generate-CTMR \
  --output /tmp/patient-high-resolution-new-run --seed 42
```

The output directory must not already exist. The recipe uses the upstream module
from its checkout working directory, with no `PYTHONPATH` changes. It performs no
model downloads. Identical seeds need not yield bit-identical output across
software/hardware versions. The recorded environment file describes this local
run; rerun the recipe to produce paths for a different machine.

## Provenance and use limitations

Source mask: `Task03/labelsTr/liver_109_133combined_aug_wbdm.nii.gz` in
NV-Generate-CT's `all_masks_flexible_size_and_spacing_4000` database. This is an
augmented/pseudo-labeled database artifact, not manually verified ground truth.
The newly generated CT does not make the source anatomy fully synthetic.

References: [NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR),
[NV-Generate-CT model/data](https://huggingface.co/nvidia/NV-Generate-CT),
[Medical Segmentation Decathlon](http://medicaldecathlon.com/).
Retain these source attributions. The repository code license does not automatically
license third-party dataset contents; review upstream model and dataset terms
before redistribution. No model weights are included here. This example is not
for diagnosis, treatment, or any clinical decision.
