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
    A --> B["HumanBody: user API + attached imaging"]
    B --> T["extract_topology: local centerline graphs"]
    CT["Native ScanVolume + full affine"] --> V["attach_scan: ImagingVolume"]
    B --> V
    B --> E["exporters"]
    T --> E
    V --> E
    E --> U["USD: anatomy + optional graphs and CT"]
    E --> P["Patient bundle: manifest + USD + optional HU and anatomy arrays"]
```

## Main entry points

- [SegmentationImporter](importers/_segmentation.py) reads labeled NIfTI,
  binary-mask directories, or NumPy masks. `to_anatomy_collection()` extracts meshes.
- [HumanBody](human.py) owns `anatomy` and optional `imaging`, and exposes
  `extract_topology()`, `attach_scan()`, `attach_imaging()`, `export_to_usd()`, and `export_patient_twin()`.
- [AnatomyCollection](anatomy.py) exposes `structures[name]` and controls visibility.
  Disabling a structure preserves its mesh.
- The optional `nvsegment` and `nvgenerate` extras enable the corresponding
  importers. Model dependencies load only when inference is requested; source
  checkouts and weights are separate. Inference runs in-process unless an
  explicit `python_executable` is provided.
- [__main__.py](__main__.py) provides the NV-Segment / NV-Generate CLI.
- [scan_volume.py](scan_volume.py) reads native NIfTI/DICOM grids and saves or replays
  NumPy + YAML artifacts. The same helper is shipped in sensor-simulation.
- [topology.py](topology.py) calculates mesh and mask centerlines;
  [artifacts.py](artifacts.py) writes native volume/mask/graph arrays.

## Coordinates and outputs

Meshes and stored structure centerlines use XYZ meters. `structure.mesh.vertices`
uses the local frame; `structure.body_vertices` includes the original body
placement. `structure.world_vertices` includes the current display placement.

`attach_scan()` retains native values and geometry for export; `attach_imaging()`
remains a lower-level KJI/RAS-meter API. With CT, schema-3 USD and bundle geometry
use the scan physical frame and units. CT and masks preserve source array order.
Simulator world placement is applied downstream.

With attached CT and `vessel_names`, bundle export calculates a navigation graph
from the retained source labels, or rasterizes meshes if labels are unavailable.
See the [export steps](../README.md#4-export) and [USD guide](../docs/usd.md).

## Small implementation, direct controls

`anatomy.py` owns structures and their collection; `human.py` owns the body and
attached imaging. Importer runtime helpers live together in `importers/_common.py`.
The bundle exporter owns scan-frame placement and shares array writing with
`artifacts.py`. `scan_volume.py` stays synchronized with sensor-simulation.

There is no YAML configuration policy or `configuration=` argument. Set
`structure.enabled` directly, or use `body.anatomy.set_structure_enabled`,
`set_system_enabled`, and `set_enabled`. Each call changes current visibility;
the last call wins. Hidden geometry remains stored for later re-enabling.

`attach_scan` shares the scan's read-only voxel buffer. Raw arrays passed to
`attach_imaging` are still copied. Existing centerline tuple return order and
bundle metadata remain compatible with workflow consumers.
