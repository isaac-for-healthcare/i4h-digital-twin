# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call upstream paired CT/mask inference; also usable by the process fallback."""


def generate(seed, mask_output, ct_output):
    import json
    import shutil
    import sys
    from pathlib import Path

    from scripts import inference, sample

    folder = Path(mask_output).parent
    environment = json.loads(Path("configs/environment_rflow-ct.json").read_text())
    config = json.loads(Path("configs/config_infer.json").read_text())
    # A size condition triggers fresh mask diffusion instead of selecting a
    # cached mask. Use the first reference condition, without resizing organs.
    conditions = json.loads(
        Path(environment["all_anatomy_size_conditions_json"]).read_text()
    )
    config.update(
        num_output_samples=1,
        controllable_anatomy_size=[["liver", conditions[0]["organ_size"][1]]],
        image_output_ext=".nii.gz",
        label_output_ext=".nii.gz",
    )
    environment["output_dir"] = str(folder / "generated")
    env_path, config_path = folder / "environment.json", folder / "inference.json"
    env_path.write_text(json.dumps(environment))
    config_path.write_text(json.dumps(config))
    # Upstream normally keeps only the conditioning organ in its saved mask.
    # Preserve the complete generated segmentation for anatomy import.
    original_filter = sample.filter_mask_with_organs
    original_argv = sys.argv
    sample.filter_mask_with_organs = lambda labels, organs: labels
    sys.argv = [
        "inference",
        "-t",
        "configs/config_network_rflow.json",
        "-e",
        str(env_path),
        "-i",
        str(config_path),
        "--random-seed",
        str(seed),
        "--version",
        "rflow-ct",
    ]
    try:
        inference.main()
    finally:
        sample.filter_mask_with_organs = original_filter
        sys.argv = original_argv
    images = list((folder / "generated").glob("*_image.nii.gz"))
    if len(images) != 1:
        raise RuntimeError(f"Expected one generated CT, found {len(images)}")
    mask = images[0].with_name(images[0].name.replace("_image.nii.gz", "_label.nii.gz"))
    if not mask.is_file():
        raise RuntimeError("Generated CT is missing its paired segmentation")
    shutil.copyfile(mask, mask_output)
    shutil.copyfile(images[0], ct_output)


if __name__ == "__main__":
    import sys

    generate(*sys.argv[1:])
