# Bundled imaging-to-mesh subpackage

Install the patient package; no separate meshing distribution or source-path
configuration is needed. Run these install commands from the repository root:

```bash
pip install ./patient-digital-twin
# For an editable installation:
pip install -e './patient-digital-twin'
```

```python
import numpy as np
from patient_digital_twin.imaging_to_mesh import mask_to_mesh

mask = np.zeros((16, 16, 16), dtype=bool)  # Z, Y, X
mask[4:12, 4:12, 4:12] = True
vertices_xyz_mm, faces = mask_to_mesh(
    mask, spacing_zyx_mm=(2.0, 1.0, 1.0), origin_xyz_mm=(0.0, 0.0, 0.0)
)
```

Only NumPy and scikit-image are needed for this operation; both are patient
package dependencies. Padding produces closed surfaces at scan boundaries.
Vertices are XYZ float32 coordinates and faces are int32 triangle indices.

For NIfTI labels, use `SegmentationImporter.to_anatomy_collection()`: it applies the
full imaging affine and converts to meters, rather than using only spacing
and origin. See the [patient API guide](../README.md).
