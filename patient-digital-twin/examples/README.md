# HumanBody pipeline

Run from `patient-digital-twin/`. Install `.[usd]` and the mesh/topology dependencies
(`scipy`, `vtk`; add `trimesh` for STL/OBJ import). Install `.[soma]` only when
attaching an external body.
Inference backends require their own models and dependencies; NVIDIA backends can
run in a separate interpreter using `--python`.

The pipeline imports anatomy, constructs `HumanBody`, attaches matching CT when
available, extracts vessel/airway topology, optionally attaches SOMA, then calls
`body.export_patient_twin()` and `body.export_to_usd()`. Only those exporters write
the output assets. Failure leaves no partially exported output directory.

| Source | Anatomy | CT attachment | SOMA |
| --- | --- | --- | --- |
| `nvgenerate` | Fresh generated segmentation → meshes | Matching generated CT | Optional `--soma` |
| `nvsegment` / `totalsegmentator` | Input CT → segmentation → meshes | The input CT | Optional `--soma` |
| `simple` | Supplied STL/OBJ/USD meshes | Skipped | Skipped |
| `sample` / `segmentation` | Existing segmentation → meshes | Matching sample CT / required `--ct` | Optional `--soma` |

```bash
python examples/pipeline.py --source nvgenerate \
  --source-root /path/to/NV-Generate-CTMR --output /tmp/generated-patient
python examples/pipeline.py --source nvsegment --input /path/to/ct.nii.gz \
  --bundle-root /path/to/NV-Segment-CTMR --output /tmp/segmented-patient
python examples/pipeline.py --source totalsegmentator --input /path/to/ct.nii.gz \
  --soma --output /tmp/patient-with-skin
python examples/pipeline.py --source simple --input /path/to/meshes.json \
  --output /tmp/mesh-patient
python examples/pipeline.py --source sample --output /tmp/sample-patient
```

NVGenerate uses upstream paired `rflow-ct` inference and retains all generated
labels, rather than only the organ used to condition generation. It needs both
mask and image-generation checkpoints and the anatomy-size conditioning dataset.
The importer retains the matching NumPy CT and voxel affine long enough to attach
them to the body; it does not substitute another scan or perform SOMA fitting.

Simple input JSON maps anatomy names to mesh files relative to the JSON:

```json
{"meshes": {"liver": "liver.stl", "aorta": "aorta.stl"}}
```

Meshes use XYZ meters in a shared body frame. Optional `mesh_to_body` matrices
place individual local meshes. Omitted matrices are identity in this pipeline;
no sample-patient placement is inferred. `--ct`, `--soma`, and `--parameters` are
rejected for simple inputs.

Use `--anatomy aorta liver` to select exported structures, or omit it for all
imported anatomy. Disabled meshes remain present but invisible. Topology is
extracted before optional SOMA attachment, in each structure's local coordinates.
`--centerline-spacing-mm` controls its voxel grid. Extraction failures stop export.
SOMA uses all available bones for matching before the export subset is applied.
Partial scans may need `--soma --parameters /path/to/registration.json`.

The output directory must be new and contains:

| Output | Contents |
| --- | --- |
| `human_body.usdc` | All anatomy meshes, embedded centerlines, attached CT attributes, optional SOMA skin |
| `patient_twin.yaml` | Bundle inventory, coordinates, mesh paths, and centerline asset paths |
| `patient_anatomy.usdc` | Bundle anatomy, embedded centerlines, optional SOMA exterior; CT is stored in external arrays |
| `centerlines/*.npz` | Local-meter points, edges and radii; transforms recorded in the manifest |
| `hu_volume.npy`, `mu_volume.npy`, `metadata.json` | Attached CT and attenuation, when CT exists |

With no source registration, a mesh-only bundle explicitly uses the `body` frame.
Registered bundles use DICOM LPS. No exterior is fabricated when SOMA is absent.
The bundle export API still offers explicit CT-envelope and composite navigation
exports through `exterior="ct"` and `vessel_names=...`; the pipeline does not require
them. Physics-demo exports are separate from this pipeline.

## Isaac Sim viewer

```bash
/path/to/isaac-sim/python.sh examples/isaac_sim.py view /tmp/mesh-patient/human_body.usdc
```

The loader opens the USD, adds session-only lighting, and frames the patient.
Select structures under `/HumanBody/Anatomy`; when present, hide
`/HumanBody/Exterior` to inspect internal anatomy. Imaging attributes store data;
they do not provide volume rendering. USD is a static snapshot.

For the separate minimal example that always attaches SOMA to the sample:

```bash
python examples/isaac_sim.py export /tmp/human_body.usdc
/path/to/isaac-sim/python.sh examples/isaac_sim.py view /tmp/human_body.usdc
```

`viewer.py` remains an optional browser-based pose diagnostic. Its `--bundle`
argument reads legacy `body.json`/`body.npz` caches; this pipeline produces USD and
patient bundles for Isaac Sim instead of those caches.

See [Working with patient USD files](../docs/usd.md) for programmatic inspection,
coordinate handling, and the differences between the two USD outputs.
