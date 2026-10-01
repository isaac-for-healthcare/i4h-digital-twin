# Isolated CT artifact compatibility component

This temporary module contains the CT ingest, LPS orientation, HU-to-mu mapping,
and volume-cache components from repository main at
`d32b35170ea5ed3a9c3e521929469d9d98be2244`, plus the mask-to-centerline routine
from its vessel artifact CLI. It does not include a segmentation backend or
simulation code. Its imports stay within this module, NumPy, and optional IO or
morphology dependencies so it can be removed or replaced independently.

Run the component without the patient inference pipeline:

```bash
python -m patient_digital_twin.legacy_ct \
  --ct /path/to/ct.nii.gz --vessel-mask /path/to/vessels.nii.gz \
  --output /tmp/new-ct-artifacts
```

The mask must share the CT's physical grid. CT-only use omits `--vessel-mask`.
The output includes `mu_volume.npy`, `metadata.json`, and `hu_volume.npy`.
With a vessel mask it also includes `vessel_mask.npy`, `centerline_points_mm.npy`,
`centerline_edges.npy`, and `centerline_radii_mm.npy`. Graph positions and radii
use patient LPS millimeters; arrays use ZYX order. Oblique grids are rejected.

Python callers can use `write_artifacts(ct, output, vessel_mask=mask)` or the
ported `VolumePreprocessor` and `HuToMuMapping` APIs. `write_artifacts` accepts an
existing `(points_mm, edges, radii_mm)` graph; otherwise it skeletonizes the mask.
The `interventional` attenuation preset is the workflow default; `linear` retains
the original ramp. This component does not write a patient manifest or USD;
those remain owned by the patient exporter.
