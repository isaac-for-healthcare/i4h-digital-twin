# Isolated CT imaging and navigation component

This temporary component retains canonical LPS CT ingest, spatial metadata, and
mask-to-centerline processing from the legacy patient implementation. X-ray
attenuation conversion and presets now belong to `xray_simulator` in
**i4h-sensor-simulation**; this module has no dependency on that renderer.

```bash
python -m patient_digital_twin.legacy_ct \
  --ct /path/to/ct.nii.gz --vessel-mask /path/to/vessels.nii.gz \
  --output /tmp/new-ct-artifacts
```

The mask must share the CT's physical grid. Omit it for CT-only export.
The output contains `hu_volume.npy` and `metadata.json` with array order, HU units,
spacing, origin, direction and source orientation. With a mask, it also includes
`vessel_mask.npy`, `centerline_points_mm.npy`, `centerline_edges.npy`, and
`centerline_radii_mm.npy`. Volumes use ZYX order; points/radii use LPS millimeters.
Oblique grids are rejected; no intensity clipping or HU-to-μ conversion occurs.

Python callers use `write_artifacts(ct, output, vessel_mask=mask)`. An optional
`centerline=(points_mm, edges, radii_mm)` supplies an existing graph; otherwise the
mask is skeletonized. This standalone component writes neither USD nor a patient
manifest; the patient exporter adds those.
