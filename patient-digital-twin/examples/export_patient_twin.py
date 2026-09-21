# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CT → NV-Segment HumanBody → i4h-workflows patient_twin.yaml bundle."""

import argparse
import json
from pathlib import Path

from patient_digital_twin.importers import NVSegmentImporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct",
        type=Path,
        default=Path("~/dev/data/Totalsegmentator_dataset_small_v201/s0011/ct.nii.gz"),
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--soma-parameters",
        type=Path,
        help="Optional scan registration parameters as JSON",
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    body = NVSegmentImporter(
        args.ct.expanduser(), bundle_root=args.bundle_root
    ).to_human_body()
    parameters = (
        json.loads(args.soma_parameters.read_text()) if args.soma_parameters else {}
    )
    body.attach_soma(**parameters)
    print("SOMA landmark residuals (m):", body.soma.alignment_errors_m, flush=True)
    print(
        body.export_patient_twin(args.output, ct_path=args.ct, exterior="soma"),
        flush=True,
    )


if __name__ == "__main__":
    main()
