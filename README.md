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
| [`patient_digital_twin.exporters`](./patient-digital-twin/patient_digital_twin/exporters/) | Bundled patient subpackage | USD, workflow bundles, CT attenuation, physics demo exports |
| [`patient_digital_twin.imaging_to_mesh`](./patient-digital-twin/patient_digital_twin/imaging_to_mesh/README.md) | Bundled patient subpackage | Binary masks → NumPy vertices and faces |
| [`patient_digital_twin.importers`](./patient-digital-twin/examples/README.md) | Bundled patient subpackage | Synthetic CT/MR masks, segmentation and mesh import |

### Quick start — installable packages

```bash
# Patient anatomy and bundled meshing
cd patient-digital-twin
uv sync --extra dev --extra usd
```

Python API example (segmentation → patient anatomy meshes):

```python
from patient_digital_twin import HumanBody, SegmentationImporter

anatomy = SegmentationImporter("patient_label.nii.gz", "label_dict.json").to_anatomy_collection()
body = HumanBody(anatomy)
for structure in body.anatomy.select(include_empty=False):
    print(structure.name, structure.world_vertices.shape, structure.faces.shape)
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
| Python | 3.10+ for installable packages (`patient_digital_twin` and the other twin modules) |
| GPU | Optional for NV-Segment / NV-Generate / Isaac Sim; CPU paths exist for basic vessel masking and mesh conversion |
| Tooling | `uv` or `pip`; Isaac Sim when loading USD in simulation |

Installable packages do **not** require Conda. Hospital / robot twin guides may assume Isaac Sim, Isaac Lab, or XR runtimes — see each component README.

## Python Packages

Install every top-level digital twin module from the repository root:

```bash
uv pip install .
# or: pip install .
```

Install one top-level module by supplying its directory instead:

```bash
uv pip install ./patient-digital-twin
uv pip install ./hospital-digital-twin
uv pip install ./sim-ready-assets
uv pip install ./robot-digital-twin
```

The corresponding Python imports are `patient_digital_twin`, `hospital_digital_twin`, `sim_ready_assets`, and `robot_digital_twin`.

To install patient USD support in the repository's own `.venv`, run from this root:

```bash
uv sync --extra dev --extra patient-usd
```

## Development / CI

Installable packages under `patient-digital-twin/` include unit tests and can be exercised with:

```bash
cd patient-digital-twin
uv run --extra dev pytest
```

Repository GitHub Actions cover copyright headers, markdown link checks, pre-commit linting, and package build/test for the installable modules.

## Security

See [SECURITY.md](./SECURITY.md). Do not report security vulnerabilities through public GitHub issues.

## Support

This repository is under active development (experimental). For questions and support, open an issue in the GitHub repository.

## License

Licensing varies by component and asset source. Check each package or asset directory for SPDX / LICENSE files. Lightwheel SimReady assets under `sim-ready-assets/` are for non-commercial R&D use only unless otherwise stated.
