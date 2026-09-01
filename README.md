# Isaac for Healthcare — Digital Twin

Pipelines and packages for building patient, hospital, and robot digital twins for NVIDIA Isaac Sim and healthcare robotics.

This repository turns medical imaging, robot descriptions, and hospital scenes into simulation-ready OpenUSD assets and preprocessing artifacts that feed Isaac Lab / IsaacLab-Arena workflows.

## Repository Layout

```text
i4h-digital-twin/
├── patient-digital-twin/     # Imaging → anatomy twin (CT, vessels, USD)
├── hospital-digital-twin/    # Hospital scene authoring, teleop, sim2real
├── robot-digital-twin/       # Bring-your-own robot (CAD / URDF → USD)
└── sim-ready-assets/         # Asset catalog helper for i4h sim assets
```

## Patient Digital Twin

Convert clinical or synthetic imaging into vessel/anatomy artifacts and OpenUSD meshes.

| Component | Status | Purpose |
| --- | --- | --- |
| [`vasculature_digital_twin`](./patient-digital-twin/vasculature_digital_twin/README.md) | Installable package | CT ingest, HU→μ preprocessing, vessel masks, centerlines |
| [`imaging_to_mesh`](./patient-digital-twin/imaging_to_mesh/README.md) | Installable package | Labelmaps / NumPy masks → OBJ + OpenUSD |
| [`generate_imaging`](./patient-digital-twin/generate_imaging/README.md) | Guide | Synthetic CT/MR generation with MAISI |

### Quick start — installable packages

```bash
# CT preprocessing + vessel extraction
cd patient-digital-twin/vasculature_digital_twin
uv venv && uv pip install -e ".[dev]"
vdt-preprocess-ct --nifti /path/to/ct.nii.gz --output-dir /tmp/ct_cache
vdt-segment-vessels --ct-dir /tmp/ct_cache --no-totalsegmentator

# Mask / labelmap → USD
cd ../imaging_to_mesh
uv venv && uv pip install -e ".[dev]"
imaging-to-mesh /path/to/patient_label.nii.gz --output-dir /tmp/usd_out
```

Python API example (mask from vasculature twin → USD):

```python
from imaging_to_mesh import convert_mask_to_usd
from vasculature_digital_twin import VolumePreprocessor, get_vessel_mask

pre = VolumePreprocessor.from_nifti("ct.nii.gz")
volume = pre.preprocess(output_dir="ct_cache")
mask = get_vessel_mask(
    hu_zyx=pre.hu_volume_zyx,
    spacing_zyx_mm=volume.spacing_zyx_mm,
    use_totalsegmentator=False,
).combined_mask

result = convert_mask_to_usd(
    mask,
    "output/vasculature.usd",
    name="Vasculature",
    spacing_zyx_mm=volume.spacing_zyx_mm,
)
print(result.usd_path)
```

## Hospital Digital Twin

Tools for authoring hospital simulation environments and collecting / augmenting robot demonstration data.

| Component | Purpose |
| --- | --- |
| [`setup_from_assets`](./hospital-digital-twin/setup_from_assets/README.md) | Assemble operating-room scenes from catalog assets |
| [`reconstruct_from_video`](./hospital-digital-twin/reconstruct_from_video/README.md) | NuRec / neural reconstruction of hospital spaces |

## Robot Digital Twin

Bring-your-own-robot guides for converting CAD / URDF into articulated USD assets and integrating them into Isaac Sim scenes.

See [`robot-digital-twin/README.md`](./robot-digital-twin/README.md).

## Sim-Ready Assets

Catalog of robots, anatomy, equipment, and hospital environments used across i4h simulations, plus the `i4h_asset_helper` download helper.

See [`sim-ready-assets/README.md`](./sim-ready-assets/README.md).

## Requirements

Shared / typical prerequisites (exact versions depend on the component):

| Requirement | Notes |
| --- | --- |
| OS | Linux (x86_64) recommended |
| Python | 3.10+ for installable packages (`vasculature_digital_twin`, `imaging_to_mesh`) |
| GPU | Optional for TotalSegmentator / MAISI / Isaac Sim; CPU paths exist for basic vessel masking and mesh conversion |
| Tooling | `uv` or `pip`; Isaac Sim when loading USD in simulation |

Installable packages do **not** require Conda. Hospital / robot twin guides may assume Isaac Sim, Isaac Lab, or XR runtimes — see each component README.

## Development / CI

Installable packages under `patient-digital-twin/` include unit tests and can be exercised with:

```bash
cd patient-digital-twin/vasculature_digital_twin && uv pip install -e ".[dev]" && pytest
cd ../imaging_to_mesh && uv sync --extra dev && uv run pytest
```

Repository GitHub Actions cover copyright headers, markdown link checks, pre-commit linting, and package build/test for the installable modules.

## Security

See [SECURITY.md](./SECURITY.md). Do not report security vulnerabilities through public GitHub issues.

## Support

This repository is under active development (experimental). For questions and support, open an issue in the GitHub repository.

## License

Licensing varies by component and asset source. Check each package or asset directory for SPDX / LICENSE files. Lightwheel SimReady assets under `sim-ready-assets/` are for non-commercial R&D use only unless otherwise stated.
