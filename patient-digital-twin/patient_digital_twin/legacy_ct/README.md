# Isolated CT imaging and navigation component

This temporary component retains the skeleton-centerline algorithm from the legacy
implementation. HU → μ conversion belongs to sensor-simulation.

```bash
python -m patient_digital_twin.legacy_ct \
  --ct /path/to/ct.nii.gz --vessel-mask /path/to/vessels.nii.gz \
  --output /tmp/new-ct-artifacts
```

The mask must match the CT grid. Output contains native `volume.npy` in HU and
`volume.yaml`; with a mask it adds `vessel_mask.npy`, `centerline_points.npy`,
`centerline_edges.npy`, and `centerline_radii.npy`. Points and radii use the source
scan frame and units. Rotated orthogonal grids are supported without resampling.
No USD or patient manifest is written by this standalone command.

The older Python `write_artifacts`/canonical-ingest APIs remain isolated for legacy
callers; current bundle exports use `scan_volume` and native-grid helpers instead.
