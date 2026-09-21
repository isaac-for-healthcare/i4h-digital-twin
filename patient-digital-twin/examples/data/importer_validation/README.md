# Real importer and viewer validation

Validated on 2026-09-18 using the actual inference backends, the existing SOMA
implementation, and `HumanBodyViewer`. Both segmentation runs used the bundled
`nv_ct_high_resolution/ct.nii.gz`. NV-Generate sampled a fresh mask with seed
277571733; a separate run with another seed also completed. The simple example
loaded `../colon.stl` with its saved SOMA parameters.

The `*_report.json` files record available and missing catalog anatomy. The
`*_viewer.json` files record every expected/visible mesh and full vertex plus
face-center containment at the initial scan pose. These are actual scene checks,
not browser screenshot checks, and do not establish containment at other poses.
Real MR inference was not run; modality routing is covered by adapter tests.

NV-Generate uses preview registration from its two available hip landmarks and
an assumed superior direction. Its limited field of view cannot support the
usual shoulder-and-hip registration. The other inferred bodies use bone landmark
matching. Optional fitting adjusts the existing SOMA skin, keeping anatomy rigid.
A visible mesh does not imply that it is fully enclosed by the skin.

All three inference backends lack five catalog labels in the selected tasks:
`intervertebral_discs`, `lung_left`, `lung_right`, `vertebrae`, and `vertebrae_L6`.
Individual lung lobes and vertebrae remain available. Additional absent labels
are listed separately in each report. No anatomy is fabricated to fill gaps.

Validation used TotalSegmentator 2.18.0, the NV-Segment CT_BODY MONAI bundle,
and the NV-Generate native mask DDPM sampler. NVIDIA workflows require source
checkouts plus their optional Python requirements and pretrained weights.

## Results

| Importer | Visible meshes | Fully contained meshes | Maximum protrusion (mm) |
| --- | ---: | ---: | ---: |
| simple | 1/1 | 1/1 | 0.00 |
| nvgenerate | 60/60 | 44/60 | 18.68 |
| nvsegment | 112/112 | 110/112 | 0.94 |
| totalsegmentator | 109/109 | 107/109 | 2.29 |

The CT adapters used 12 shape components and up to 12 fitting iterations.
The generated-mask fit used 12 components and stopped improving after six
iterations. The containment requirement is **not fully satisfied** by the three
inferred examples. The stored JSON identifies every protruding structure; no
mesh was hidden or deformed to make these checks pass.
