# HumanBody importers

The four adapters live in `patient_digital_twin.importers`. Each has
`to_human_body()` and a `.report` containing `present`, `absent`, and
`unsupported` names. All supported catalog structures are requested; unavailable
anatomy stays empty. A partial scan cannot contain anatomy outside its field of
view. Segmentation labels and generated masks are not assumed to be ground truth.

The shared `importers/_segmentation.py` implementation handles mask-to-mesh
conversion. `SegmentationImporter` is available from `patient_digital_twin.importers`
and remains exported from `patient_digital_twin` for existing callers. No anatomy, SOMA, fitting, or viewer implementation changes are needed.

## Optional installations

The base package does not install these runtimes or download their weights.
Adapters raise an installation error when the required runtime is missing.

```bash
# TotalSegmentator's real Python package:
pip install TotalSegmentator
# STL/OBJ and OpenUSD input (install only what you use):
pip install trimesh usd-core
```

NV-Generate and NV-Segment are **source workflows**, not installable packages
named `nvgenerate` or `nvsegment`. Follow their upstream installation instructions,
pip-install the optional runtime dependencies, and supply the checkout path:

- [NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR):
  `pip install -r /path/to/NV-Generate-CTMR/requirements.txt` in its environment.
  Download `models/mask_generation_autoencoder.pt`,
  `models/mask_generation_diffusion_unet.pt`, and
  `datasets/all_anatomy_size_conditions.json` using upstream setup. The local
  `temp_work_dir/datasets` layout is also supported.
- [NV-Segment-CTMR](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR):
  the adapter uses its inner `NV-Segment-CTMR/configs/inference.json` MONAI bundle.
  Its inference runtime can be installed with
  `pip install torch 'monai>=1.5' pytorch-ignite fire einops huggingface_hub`.
  Older upstream requirements pin Python/NumPy versions; use a compatible
  environment instead of installing those pins into a modern environment.
  The upstream bundle downloads its checkpoint on first use.
- [TotalSegmentator Python API](https://github.com/wasserth/TotalSegmentator#python-api):
  weights are downloaded on first use. CT uses `total`; MR uses `total_mr`.
  Specialty/licensed models are not silently enabled to fill coverage gaps.

The NVIDIA adapters accept `python_executable` to use a separate environment.
Their upstream scripts run in a child process so the two different `scripts`
packages do not collide in the application interpreter. NV-Generate requires
CUDA. For viewer examples also install this project's optional SOMA/viewer extras
and SOMA assets as described in the existing package README.

## Usage

```python
from patient_digital_twin.importers import (
    NVGenerateImporter, NVSegmentImporter, TotalSegmentatorImporter, SimpleImporter,
)

body = NVGenerateImporter(source_root="/path/to/NV-Generate-CTMR").to_human_body()
body = NVSegmentImporter("ct.nii.gz", bundle_root="/path/to/NV-Segment-CTMR/NV-Segment-CTMR").to_human_body()
body = TotalSegmentatorImporter("mr.nii.gz", modality="MR").to_human_body()
body = SimpleImporter({"colon": "examples/data/colon.stl"}).to_human_body()
```

Image inputs may be a NIfTI path, a nibabel NIfTI image, or a **XYZ NumPy array**.
An array must also supply `affine_xyz_to_imaging_m`; voxel spacing and orientation
cannot be inferred from its shape. File/header units are converted to millimeters
for backend inference and to meters for HumanBody meshes. For example:

```python
import numpy as np
importer = TotalSegmentatorImporter(
    np.zeros((128, 128, 128), dtype=np.float32),
    affine_xyz_to_imaging_m=np.diag([0.001, 0.001, 0.002, 1]),
)
# Replace zeros with the actual CT intensities before running inference.
```

NV-Generate invokes only the upstream mask diffusion sampler, with its native
256³ / 1.5 mm grid, 1000 DDPM steps, and default sampler decode settings. With no
organ-size override it uses the upstream conditioning table's first record,
matching the upstream conditioning selector's tie-break. Every call uses a new
random seed, recorded in `.seed` and `.report`. It does not run CT synthesis or
reuse a database mask. The model often produces a partial torso/pelvis, not an
entire head-to-toe body. Generating a new mask does not guarantee every class.

STL/OBJ coordinates are **local XYZ meters**. USD uses authored stage units and
transforms; Y-up is converted to Z-up. Supply one anatomy-to-file entry per
structure. USD files must contain triangle meshes. Explicit `mesh_to_body`
matrices are proper rigid transforms from file coordinates to the body frame:

```python
placement = np.eye(4)
placement[:3, 3] = [0.01, 0.02, 0.03]
body = SimpleImporter(
    {"colon": "colon.obj"}, mesh_to_body={"colon": placement},
).to_human_body()
```

Omitted transforms use `importers/reference_body.json`, derived from the bundled
high-resolution segmentation. Its origin is the anatomy bounding-box center.
Default structure centers are sample bounding-box centers; the bundled colon's
placement uses its extracted surface centroid. These are **reference placements**
for centered local meshes, not automatic registration of arbitrary new meshes.
No input mesh is automatically resized. Supplying every transform avoids the
reference dependency and does not infer an imaging relationship. The optional
`body_to_imaging` preserves a caller-provided relationship to an imaging frame.

## Runnable viewer examples

From `patient-digital-twin/`, using an environment with the viewer extras:

```bash
python examples/simple_importer.py
python examples/nvgenerate_importer.py --source-root /path/to/NV-Generate-CTMR --python /path/to/generator/python
python examples/nvsegment_importer.py --bundle-root /path/to/NV-Segment-CTMR/NV-Segment-CTMR
python examples/totalsegmentor_importer.py
```

The segmenter examples default to `examples/data/nv_ct_high_resolution/ct.nii.gz`.
Use `--image` and `--modality MR` for another scan. The simple example loads
`examples/data/colon.stl` and its fitted external-body parameters in
`colon_soma_parameters.json`. Rebuild the STL and reference placements with
`python examples/prepare_simple_importer_data.py`.

Every example accepts `--validate-only --report result.json`. The report separates
catalog coverage, viewer visibility, alignment residuals, and full vertex/face-center
containment. Validation exits nonzero when any imported mesh protrudes. Missing
catalog meshes are reported, not drawn as fabricated anatomy. All anatomy remains
rigid, so containment is not guaranteed for an arbitrary patient or pose.

Use `--parameters registered.json` to supply measured body-frame landmarks or
an explicit `body_to_soma` transform. NV-Generate's default example enables
`--partial-preview`: when only the two hip landmarks are available, it aligns
them and assumes imaging RAS +Z is superior. This is explicitly labeled a preview
registration in the report. It cannot establish accurate pose from two points.
Other examples require explicit parameters if fewer than three shoulder/hip
landmarks are available. `--fit-shape` runs the existing bounded SOMA shape fitter
at the scan pose; it does not deform anatomy or guarantee containment.

## Adapter checks

```bash
pytest examples/test_importers.py
```

These tests cover input units, backend label IDs, modality selection, changing
seeds, missing optional dependencies, STL/OBJ placement, and USD authored units.
Mocked backend tests do not establish segmentation accuracy. Real-run coverage
and viewer/containment results are recorded in `examples/data/importer_validation/`.

## Independent CT-to-centerline comparison

`i4h_workflows_demo.py` starts two independent pipelines from the same CT file:

1. `vasculature_digital_twin_pipeline(ct_path)` runs TotalSegmentator using the
   vasculature package, then creates its centerline graph using the voxel
   skeletonization used by i4h-workflows.
2. `patient_digital_twin_pipeline(ct_path)` runs `NVSegmentImporter` to create a
   `HumanBody`, voxelizes its independently predicted aorta and iliac meshes,
   and stores the composite tree's centerline through `body.extract_topology()`.

Neither pipeline reads supplied segmentation files or uses the other pipeline's
masks, meshes, or centerlines. Both independently apply the same fixed union,
closing (two iterations), largest-component selection, and skeletonization.
Coordinates are compared in LPS millimeters. The patient graph is stored locally
in meters, with its composite mesh placed in the imported HumanBody frame.

Install the local `vasculature-digital-twin` and `i4h-tools-patient-twin` packages,
TotalSegmentator, the NV-Segment runtime, VTK, and matplotlib. The workflow package
specifies Python 3.12. From `patient-digital-twin/`:

```bash
python examples/i4h_workflows_demo.py \
  --ct ~/dev/data/Totalsegmentator_dataset_small_v201/s0011/ct.nii.gz \
  --bundle-root /path/to/NV-Segment-CTMR/NV-Segment-CTMR
```

`--python` selects a separate NV-Segment interpreter. The default `--stage both`
runs both inference pipelines anew. `--stage vasculature` and `--stage patient`
run independently; `--stage compare` compares their saved results after verifying
that both CT hashes match. Artifacts are saved to
`examples/data/i4h_workflows_s0011_independent/` by default: separate graphs,
masks and provenance, a numeric comparison, and a side-by-side CT overlay.

Independent models need not produce equal nodes or connectivity. Comparison
samples each graph's edges every 0.5 mm and reports symmetric nearest-centerline
mean, 95th percentile, maximum distance, bidirectional coverage, and segmentation
Dice. The preselected example criterion is at least 95% coverage in **both**
directions within 3 mm. CLI options expose these tolerances; they are engineering
checks, not clinical acceptance criteria. Failure saves the evidence and exits
nonzero. This replaces the earlier shared-segmentation round-trip comparison.
