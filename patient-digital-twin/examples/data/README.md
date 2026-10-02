# Example CT and segmentations

`s0011/` contains the unmodified CT and named binary masks from Jakob Wasserthal's
[TotalSegmentator small dataset v2.0.1](https://zenodo.org/records/10047263),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`provenance.json` records attribution and SHA-256 hashes. These masks are supplied
dataset annotations; the NVSegment example computes a new segmentation from CT.

The 46.7 MB `s0011/ct.nii.gz` uses Git LFS. From the repository root:

```bash
git lfs install
git lfs pull --include="patient-digital-twin/examples/data/s0011/ct.nii.gz"
```

Each file in `s0011/segmentations/` is a binary NIfTI named for its structure.
Examples select the aorta by default. See the [example commands](../README.md).
