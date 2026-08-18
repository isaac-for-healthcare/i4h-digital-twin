# Imaging to Mesh

Installable Python package for converting medical segmentation labelmaps and
NumPy masks into OBJ meshes and OpenUSD stages.

## Install

From this package directory:

```bash
uv venv
uv pip install -e .
```

From another application:

```bash
python -m pip install /path/to/i4h-digital-twin/patient-digital-twin/imaging_to_mesh
```

NRRD input is optional:

```bash
python -m pip install "imaging-to-mesh[nrrd]"
```

The base package supports Python 3.10+, NIfTI, NumPy arrays, OBJ, and OpenUSD.
It does not require Conda, MONAI, VTK, or Isaac Sim.

## Python API

Convert a binary NumPy mask, such as a vessel mask produced by
`vasculature_digital_twin`:

```python
from imaging_to_mesh import convert_mask_to_usd

result = convert_mask_to_usd(
    vessel_mask,
    "output/vasculature.usd",
    name="Vasculature",
    spacing_zyx_mm=(1.0, 0.8, 0.8),
    origin_xyz_mm=(0.0, 0.0, 0.0),
)
print(result.usd_path)
```

Convert a MAISI/TotalSegmentator NIfTI labelmap:

```python
from imaging_to_mesh import convert_segmentation_file

result = convert_segmentation_file(
    "patient_label.nii.gz",
    "output/patient",
)
for mesh in result.meshes:
    print(mesh.name, mesh.obj_path)
print(result.usd_path)
```

Lower-level reusable components are also public:

- `load_labelmap`
- `mask_to_mesh`
- `write_obj`
- `write_usd`
- `convert_segmentation_array`
- `convert_nrrd_to_nifti`

## Command Line

```bash
imaging-to-mesh patient_label.nii.gz --output-dir output/patient
```

For a directory:

```bash
imaging-to-mesh /path/to/scans \
  --pattern '.*label\.(nii(\.gz)?|nrrd)$' \
  --output-dir output
```

## Output

```text
output/
├── obj/
│   ├── Liver.obj
│   ├── Veins.obj
│   └── ...
└── all_organs.usd
```

USD stages use millimeters and set `metersPerUnit` to `0.001`.

## Development

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
uv build
```
