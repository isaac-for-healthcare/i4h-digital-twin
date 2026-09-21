# Independent CT-to-centerline validation

Ran both model pipelines on s0011's `ct.nii.gz` on 2026-09-18. No supplied
segmentation files, reference masks, or shared predictions were used.

- Vasculature pipeline: actual TotalSegmentator `total` inference through
  `vasculature_digital_twin`, followed by the workflow's voxel centerline algorithm.
- Patient pipeline: actual `NVSegmentImporter` CT_BODY inference, yielding a
  HumanBody with 109 nonempty catalog structures. Its aorta and iliac meshes were
  voxelized independently, combined, and extracted through `HumanBody.extract_topology`.

The fixed comparison region is the aorta plus left/right iliac arteries. Both
pipelines independently apply two closing iterations and largest-component
selection. Agreement here is for this region and subject, not all vessels or scans.

| Measurement | Result |
| --- | ---: |
| TotalSegmentator graph | 515 points / 514 edges |
| NV-Segment graph | 514 points / 513 edges |
| Mean symmetric sampled distance | 0.4914 mm |
| 95th percentile symmetric sampled distance | 1.5000 mm |
| Maximum symmetric sampled distance | 2.5981 mm |
| Coverage within 3 mm, each direction | 100% |
| Processed vessel-mask Dice | 0.96999 |

Passed the preselected criterion: at least 95% coverage in each direction within
3 mm. This measures agreement between models, not accuracy against ground truth.
`comparison.json` contains the numeric results and both pipeline provenance records.
`side_by_side.png` shows each result on the same CT maximum projection. NPZ graphs
include points, radii, edges and processed masks; units are recorded in the matching
JSON (vasculature millimeters, patient meters), both in the canonical LPS frame.
