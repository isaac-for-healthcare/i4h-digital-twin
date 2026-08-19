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
- `vasculature_digital_twin/ct/orientation.py`
  - Reorientation of CT volumes into the canonical anatomical frame (`to_canonical_lps`, `orientation_code`, `affine_to_lps`).
- `vasculature_digital_twin/config.py`
  - Preprocessing configuration dataclasses (`PreprocessingSettings`, `HuToMuMapping`).
- `vasculature_digital_twin/hu_mapping.py`
  - Evaluation and sampling of the HU->mu transfer function (`hu_to_mu`, `hu_to_mu_curve`).
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

Tune the HU->mu ramp while preprocessing (soft tissue and contrasted vessels):

```bash
vdt-preprocess-ct --nifti /path/to/ct.nii.gz --output-dir /tmp/ct_cache --window-center 100 --window-width 800
```

Segment vessels and extract centerline graph:

```bash
vdt-segment-vessels --ct-dir /tmp/ct_cache
```

## Anatomical frame

CT series are stored in whatever slice order the acquisition produced, so a raw array axis has no fixed anatomical meaning. Both loaders resolve this at ingest using the DICOM direction cosines or the NIfTI affine, reorienting the volume into the canonical **LPS** frame:

- axis 0 (slice) increases toward the patient's **Superior** (head),
- axis 1 (row) increases toward the patient's **Posterior** (back),
- axis 2 (column) increases toward the patient's **Left**.

Mapping array axes to world axes as `(x, y, z) = (axis 2, axis 1, axis 0)` therefore gives Left / Posterior / Superior world axes, which is the DICOM patient coordinate system. NIfTI affines are RAS and are converted to LPS, so NIfTI and DICOM inputs of the same patient land in the same frame.

Reorientation is a permutation and flip of the axes, never a resample, so an oblique acquisition keeps a residual rotation and emits a warning at load time. `metadata.json` records what happened, and downstream consumers (C-arm poses, centerline coordinates, collision geometry) should read the frame rather than assume it:

| Key | Meaning |
| --- | --- |
| `anatomical_frame` | `"LPS"` when the axes were resolved; `null` means unresolved and the axes carry no anatomical meaning |
| `source_orientation` | Directions the source axes increased toward before reorientation, e.g. `"IPL"` |
| `direction_row_major_3x3` | Direction cosines after reorientation, columns in index order `(i, j, k)`; the identity for axis-aligned inputs |

Pass `--no-reorient` (or `reorient=False` to the loaders) to inspect a series exactly as stored. Volumes built with `from_numpy` are unresolved unless the caller declares `anatomical_frame="LPS"`.

## HU to mu transfer function

`HuToMuMapping` is a piecewise-linear curve from Hounsfield Units to linear attenuation coefficients (mm^-1), clamped outside its outermost control points. With the default two points it is a single ramp, parameterized the same way as window/level on a radiology viewer: a narrower window is a steeper ramp and therefore more contrast between soft tissue and vessels.

```python
from vasculature_digital_twin import HuToMuMapping, VolumePreprocessor, PreprocessingSettings

mapping = HuToMuMapping.from_window_level(window_center=100.0, window_width=800.0)
preprocessor = VolumePreprocessor.from_nifti("/path/to/ct.nii.gz", settings=PreprocessingSettings(hu_to_mu=mapping))
volume = preprocessor.preprocess()

# Sweep the window without reloading the CT.
for width in (400.0, 800.0, 2000.0):
    rewindowed = preprocessor.with_hu_to_mu(mapping.with_window_level(window_width=width)).preprocess()

# Interactive gestures: slide the ramp (level) or change its gradient (contrast).
brighter = mapping.shifted(-200.0)
punchier = mapping.scaled(1.5)
```

Passing more than two `control_points` gives independent slopes per HU band, at the cost of more knobs to tune:

```python
mapping = HuToMuMapping(control_points=((-1000.0, 0.0), (0.0, 0.004), (300.0, 0.012), (1500.0, 0.02)))
```

The curve that produced a cached `mu_volume.npy` is recorded under `hu_to_mu` in `metadata.json`, and `HuToMuMapping.from_dict` reads it back.

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
