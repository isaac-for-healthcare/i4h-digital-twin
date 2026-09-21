# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the bundled patient, in the viewer's scan pose, to a standalone USD."""

import argparse
import json
from pathlib import Path

from patient_digital_twin import SegmentationImporter
from viewer import pose_presets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "data/patient.usdc"
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    data = Path(__file__).parent / "data/nv_ct_high_resolution"
    body = SegmentationImporter(
        data / "segmentation.nii.gz", data / "labels.json"
    ).to_human_body(configuration=data / "anatomy.yaml")
    body.attach_soma(**json.loads((data / "viewer_parameters.json").read_text()))
    body.pose(pose_presets(body)["scan"])
    print(body.export_to_usd(args.output), flush=True)


if __name__ == "__main__":
    main()
