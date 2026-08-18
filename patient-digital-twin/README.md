# Patient Digital Twin (Bring your own patient)

The Patient Digital Twin pipeline turns clinical data (imaging, physiological) into simulation-ready 3D assets in Universal Scene Description (USD) format. This lets you build and run healthcare simulations—surgical planning, training, or AI policy evaluation—using anatomically accurate, synthetic patient representations instead of real patient data. The result is reusable, privacy-safe digital twins that integrate into Isaac Sim and downstream rendering or domain-randomization workflows.

## Pipeline Overview

The typical synthetic data generation pipeline flows from medical imaging and segmentation through 3D conversion to photorealistic rendering:

```mermaid
flowchart LR
    A("CT / MR Generation + segmentation masks") --> B("3D meshes & USD Conversion")
    B --> C("Material properties assignment")
    C --> D("Style Augmentation + Photorealistics Rendering")
```

## Available Components

1. **CT / MR Generation + segmentation masks**
    - [Generate imaging data and segmentation masks](./generate_imaging/README.md)

2. **3D meshes & USD Conversion**
    - [Convert segmentation labelmaps or NumPy masks to OBJ and OpenUSD](./imaging_to_mesh/README.md)

3. **Material properties assignment**
    - Define textures and material properties on the USD assets (coming soon)

4. **Style Augmentation + Photoreal Rendering**
    - Style augmentation and photorealistic rendering integration (coming soon)
