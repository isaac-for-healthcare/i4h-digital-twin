# Patient Digital Twin (Vasculature)

Standalone Python package for CT ingestion and vasculature preprocessing artifacts used in digital twin pipelines.

This package is independently installable and executable. It contains its own IO, preprocessing, segmentation, and CLI layers under `vasculature_digital_twin`.

## Standalone package scope

- Ingests DICOM/NIfTI CT data into a consistent in-memory representation.
- Converts HU volumes to mu volumes plus serialized metadata/cache artifacts.
- Produces vessel masks and centerline graph outputs for downstream simulation/analysis.
- Exposes CLI entry points so preprocessing pipelines can run without additional package dependencies.

## Package layout

- `vasculature_digital_twin/ct/dicom_ingest.py`
  - CT IO adapters (`load_dicom_series_hu`, `load_nifti_hu`) and `CtVolume` metadata container.
- `vasculature_digital_twin/config.py`
  - Preprocessing configuration dataclasses (`PreprocessingSettings`, `HuToMuMapping`).
- `vasculature_digital_twin/preprocessor.py`
  - `VolumePreprocessor` factories (`from_dicom`, `from_nifti`, `from_numpy`) and HU->mu conversion pipeline.
- `vasculature_digital_twin/volume.py`
  - `PreprocessedVolume` + `VolumeMetadata` cache persistence (`mu_volume.npy`, `metadata.json`).
- `vasculature_digital_twin/vasculature.py`
  - Vessel extraction/segmentation helpers, territory masks, centerline/mesh support, and flow-related utilities.
- `vasculature_digital_twin/cli/preprocess_ct.py`
  - CLI for building CT cache artifacts.
- `vasculature_digital_twin/cli/segment_vessels.py`
  - CLI for vessel segmentation and centerline extraction artifact generation.

## Install

From this package directory:

```bash
pip install -e .
```

Install with full optional extras:

```bash
pip install -e ".[all]"
```

## Runtime dependencies

- Base: `numpy`, `scipy`
- Optional extras:
  - `io`: `SimpleITK`, `nibabel`
  - `segmentation`: `scikit-image`, `totalsegmentator`, `nibabel`
  - `mesh`: `vtk`, `warp-lang`

The core package can be imported with base dependencies; specific pipeline features activate as optional libraries are installed.

## CLI usage

Preprocess CT into cache artifacts:

```bash
vdt-preprocess-ct --nifti /path/to/ct.nii.gz --output-dir /tmp/ct_cache
```

Segment vessels and extract centerline graph:

```bash
vdt-segment-vessels --ct-dir /tmp/ct_cache
```

## Output artifacts

- `mu_volume.npy`
- `metadata.json`
- `hu_volume.npy` (when enabled)
- `vessel_mask.npy`
- `centerline_points_mm.npy`
- `centerline_edges.npy`
- `centerline_radii_mm.npy`

## Inter-package integration (with `catheter-vasculature-solver`)

The solver package can directly consume digital twin artifacts for patient-specific setup:

- `centerline_points_mm.npy` -> insertion track origin/direction/length
- `vessel_mask.npy` (+ `metadata.json` spacing/origin) -> vessel collision mesh via `extract_vessel_mesh`
- `mu_volume.npy` -> optional attenuation context for imaging/visualization pipelines

Minimal handoff pattern:

```python
import json
import numpy as np
from pathlib import Path
from vasculature_digital_twin.vasculature import extract_vessel_mesh

ct_dir = Path("/tmp/ct_cache")
meta = json.loads((ct_dir / "metadata.json").read_text(encoding="utf-8"))
spacing_zyx_mm = tuple(meta["spacing_zyx_mm"])
origin_xyz_mm = tuple(meta.get("origin_xyz_mm") or (0.0, 0.0, 0.0))

pts_mm = np.load(ct_dir / "centerline_points_mm.npy")
vessel_mask = np.load(ct_dir / "vessel_mask.npy")

track_start = pts_mm[0] / 1000.0
track_dir = pts_mm[1] - pts_mm[0]
track_dir = track_dir / (np.linalg.norm(track_dir) + 1e-12)
track_length = float(np.linalg.norm((pts_mm[-1] - pts_mm[0]) / 1000.0))

vessel_mesh = extract_vessel_mesh(
    vessel_mask=vessel_mask,
    spacing_zyx_mm=spacing_zyx_mm,
    origin_xyz_mm=origin_xyz_mm,
)
```

## Notes for standalone use

- The package uses a standard `src/` layout and can be reused in external projects as a preprocessing utility.
- CLI commands are wired through package entry points, enabling reproducible non-notebook processing runs.
- The catheter solver package can consume these generated artifacts, but this package does not require the solver package to function.
