# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executed in the optional upstream environment, with its checkout as cwd."""


def main():
    """Load only mask weights and invoke the upstream DDPM sampler directly."""
    import json
    import sys
    from pathlib import Path

    import nibabel as nib
    import numpy as np
    import torch
    from monai.bundle import ConfigParser
    from monai.utils import set_determinism
    from scripts.sample_mask import ldm_conditional_sample_one_mask

    if not torch.cuda.is_available():
        raise RuntimeError("NV-Generate mask diffusion requires a CUDA GPU")
    seed, output = int(sys.argv[1]), sys.argv[2]
    set_determinism(seed=seed)
    config = json.loads(Path("configs/config_network_rflow.json").read_text())
    parser = ConfigParser(config)
    device = torch.device("cuda")
    autoencoder = (
        parser.get_parsed_content("mask_generation_autoencoder_def").to(device).eval()
    )
    diffusion = (
        parser.get_parsed_content("mask_generation_diffusion_def").to(device).eval()
    )
    autoencoder.load_state_dict(
        torch.load(
            "models/mask_generation_autoencoder.pt",
            map_location=device,
            weights_only=True,
        )
    )
    checkpoint = torch.load(
        "models/mask_generation_diffusion_unet.pt",
        map_location=device,
        weights_only=False,
    )
    diffusion.load_state_dict(checkpoint["unet_state_dict"])
    # Upstream default: no organ size overrides. Select its first conditioning
    # record (the same tie-break as prepare_anatomy_size_condition([])).
    candidates = [
        Path("datasets/all_anatomy_size_conditions.json"),
        Path("temp_work_dir/datasets/all_anatomy_size_conditions.json"),
    ]
    conditions = next((p for p in candidates if p.is_file()), None)
    if conditions is None:
        raise FileNotFoundError(
            "Download the upstream all_anatomy_size_conditions.json into datasets/"
        )
    anatomy_size = json.loads(conditions.read_text())[0]["organ_size"]
    mask = ldm_conditional_sample_one_mask(
        autoencoder,
        diffusion,
        parser.get_parsed_content("mask_generation_noise_scheduler"),
        checkpoint["scale_factor"],
        anatomy_size,
        device,
        config["mask_generation_latent_shape"],
        "configs/label_dict_124_to_132.json",
    )
    image = nib.Nifti1Image(
        mask.squeeze().cpu().numpy().astype(np.uint8), np.diag([1.5, 1.5, 1.5, 1])
    )
    image.header.set_xyzt_units("mm")
    nib.save(image, output)


if __name__ == "__main__":
    main()
