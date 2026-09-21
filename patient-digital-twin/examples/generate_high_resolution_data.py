# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a high-resolution example using an existing NV-Generate-CTMR checkout.

Run with that checkout's Python environment. The anatomy is a database mask,
not a new diffusion-generated segmentation; only the paired CT is synthesized.
No resampling or smoothing is applied to the source segmentation.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np

SOURCE = "Task03/labelsTr/liver_109_133combined_aug_wbdm.nii.gz"


def digest(path):
    """Hash an artifact in bounded memory for the provenance manifest."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    """Prepare the native-resolution mask, run upstream inference, and verify pairing.

    A fresh destination is required to avoid overwriting any existing data.
    Model checkpoints and the mask database must already exist in nv_root.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nv-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data/nv_ct_high_resolution",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root, out = args.nv_root.resolve(), args.output.resolve()
    source = (
        root
        / "temp_work_dir/datasets/all_masks_flexible_size_and_spacing_4000"
        / SOURCE
    )
    model_names = [
        "autoencoder_v1.pt",
        "diff_unet_3d_rflow-ct.pt",
        "controlnet_3d_rflow-ct.pt",
    ]
    for path in [source, *(root / "models" / n for n in model_names)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    out.mkdir(parents=True, exist_ok=False)
    image = nib.load(source)
    data = np.asanyarray(image.dataobj)
    if (
        image.shape != (512, 512, 768)
        or not np.isfinite(data).all()
        or not np.equal(data, np.rint(data)).all()
    ):
        raise ValueError("Expected a finite, integer-valued 512x512x768 source mask")
    if data.min() < 0 or data.max() > 255:
        raise ValueError("Source labels do not fit uint8")
    # Change storage only; retain every label voxel and the full source affine.
    mask = nib.Nifti1Image(data.astype(np.uint8), image.affine)
    mask.header.set_xyzt_units("mm")
    nib.save(mask, out / "segmentation.nii.gz")
    del data
    labels = json.loads((root / "configs/label_dict.json").read_text())
    labels.update({"background": 0, "body": 200})
    (out / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    # SOURCE is the same fixed mask as the bundled example. Reuse its explicit
    # partial-scan pose; do not fit full arm lengths to cropped humerus fragments.
    (out / "viewer_parameters.json").write_text(
        (
            Path(__file__).parent / "data/nv_ct_high_resolution/viewer_parameters.json"
        ).read_text()
    )
    (out / "anatomy.yaml").write_text("anatomy:\n  structures:\n    skull: false\n")
    config = json.loads(
        (root / "configs/config_infer_80g_512x512x768.json").read_text()
    )
    config["spacing"] = [float(x) for x in image.header.get_zooms()[:3]]
    config["anatomy_list"] = list(labels)
    network = json.loads((root / "configs/config_network_rflow.json").read_text())
    # The image-from-mask entry point does not apply the paired CLI's override.
    network["autoencoder_def"]["num_splits"] = config["autoencoder_tp_num_splits"]
    environment = json.loads((root / "configs/environment_rflow-ct.json").read_text())
    environment["output_dir"] = str(out)
    for name, value in [
        ("inference.json", config),
        ("network.json", network),
        ("environment.json", environment),
    ]:
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "scripts.infer_image_from_mask",
        "--mask",
        str(out / "segmentation.nii.gz"),
        "-t",
        str(out / "network.json"),
        "-i",
        str(out / "inference.json"),
        "-e",
        str(out / "environment.json"),
        "--random-seed",
        str(args.seed),
    ]
    manifest = {
        "status": "prepared",
        "generator": "https://github.com/NVIDIA-Medtech/NV-Generate-CTMR",
        "generator_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_mask": SOURCE,
        "source_mask_sha256": digest(source),
        "mask_origin": "NV-Generate-CTMR database mask, not diffusion-generated; storage converted to uint8 without resampling",
        "shape_xyz": list(image.shape),
        "spacing_xyz_mm": config["spacing"],
        "seed": args.seed,
        "command": command,
        "model_sha256": {n: digest(root / "models" / n) for n in model_names},
        "software_versions": {
            name: version(name) for name in ["torch", "monai", "numpy", "nibabel"]
        },
    }
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    print(f"Generating CT into {out}; progress is in generation.log", flush=True)
    with (out / "generation.log").open("w") as log:
        subprocess.run(
            command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    (generated,) = out.glob("*_image.nii.gz")
    ct = nib.load(generated)
    if ct.shape != mask.shape or not np.allclose(ct.affine, mask.affine, atol=1e-4):
        raise ValueError("Generated CT and segmentation grids do not match")
    # Store integral HU rather than a large float32 volume (rounding <= 0.5 HU).
    values = np.asanyarray(ct.dataobj)
    if not np.isfinite(values).all() or values.min() < -32768 or values.max() > 32767:
        raise ValueError("Generated CT cannot be represented as int16 HU")
    compact_ct = nib.Nifti1Image(np.rint(values).astype(np.int16), ct.affine)
    compact_ct.header.set_xyzt_units("mm")
    nib.save(compact_ct, out / "ct.nii.gz")
    generated.unlink()  # Discard only this run's temporary float32 output.
    manifest["status"] = "complete"
    manifest["ct_storage"] = (
        "int16 HU, rounded from generated float32 (maximum 0.5 HU rounding error)"
    )
    manifest["sha256"] = {
        n: digest(out / n) for n in ["segmentation.nii.gz", "ct.nii.gz", "labels.json"]
    }
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Generation complete; CT and mask affines match.", flush=True)


if __name__ == "__main__":
    main()
