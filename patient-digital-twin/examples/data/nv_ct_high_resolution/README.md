# Shared sample dataset

This is the single dataset for `examples/pipeline.py` and `examples/viewer.py`.

- `ct.nii.gz`: generated CT, local large file excluded from Git.
- `segmentation.nii.gz`: corresponding 512 × 512 × 768 anatomical mask.
- `labels.json`: source mask label names and IDs.
- `viewer_parameters.json`: fitted SOMA registration and arms-down scan pose.
- `anatomy.yaml`: excludes the sample's misplaced skull mesh.
- `provenance.json`: source, generation command, grid, and file hashes.

The CT was generated from an existing database mask; this sample mask was not
sampled by mask diffusion. The provenance records the exact source and generation
command. For a fresh checkout, supply the recorded CT locally or use a custom
input with the pipeline. Segmentation-only viewing does not need the CT.

Write pipeline results outside this dataset directory, for example `/tmp/patient`.
See [the examples guide](../../README.md) for commands and dependencies.
