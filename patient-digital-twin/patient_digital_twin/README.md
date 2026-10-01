# patient_digital_twin

The Python library behind Patient Digital Twin. Start with the
[quickstart](../README.md) to load a segmentation, extract meshes, optionally add
centerlines and CT, and export the result.

## How the library fits together

```mermaid
flowchart TD
    S["Segmentation + label dictionary"] --> I["SegmentationImporter"]
    I --> M["imaging_to_mesh: surface extraction"]
    M --> A["AnatomyCollection: named structures + mesh placement"]
    A --> B["HumanBody: user API"]
    B --> T["extract_topology: local centerline graphs"]
    CT["CT array + voxel affine"] --> V["AttachImaging: ImagingVolume"]
    B --> V
    B --> E["exporters"]
    T --> E
    V --> E
    E --> U["USD: anatomy + optional graphs and CT"]
    E --> P["Patient bundle: manifest + USD + optional arrays"]
```

## Main entry points

- [SegmentationImporter](importers/_segmentation.py) reads labeled NIfTI,
  binary-mask directories, or NumPy masks. `to_anatomy_collection()` extracts meshes.
- [HumanBody](human.py) owns `anatomy` and optional `imaging`, and exposes
  `extract_topology()`, `AttachImaging()`, `export_to_usd()`, and `export_patient_twin()`.
- [AnatomyCollection](anatomy.py) exposes `structures[name]` and controls visibility.
  Disabling a structure preserves its mesh.
- The optional `nvsegment` and `nvgenerate` extras enable the corresponding
  importers. Model dependencies load only when inference is requested; source
  checkouts and weights are separate. Inference runs in-process unless an
  explicit `python_executable` is provided.
- [main.py](main.py) provides the NV-Segment / NV-Generate CLI.
- [legacy_ct](legacy_ct/README.md) isolates CT attenuation and navigation artifact
  generation used by the bundle exporter.

## Coordinates and outputs

Meshes and stored structure centerlines use XYZ meters. `structure.mesh.vertices`
uses the local frame; `structure.body_vertices` includes the original body
placement. `structure.world_vertices` includes the current display placement.

`AttachImaging()` takes a ZYX volume and an XYZ-voxel-to-RAS-meter affine.
Imported anatomy retains its source registration. Standalone USD preserves
current placement; patient bundles use original imaging placement.

With attached CT and explicit `vessel_names`, bundle export calculates a
navigation graph from the final CT-grid vessel mask. These navigation arrays use
LPS millimeters; stored per-structure graphs remain in local meters. See the
[export steps](../README.md#4-export) and [USD guide](../docs/usd.md) for details.
