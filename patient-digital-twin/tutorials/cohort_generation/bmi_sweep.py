# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controlled sweep: the same generation seeds at several BMIs, with age and sex held fixed.

Patients that share a seed start NV-Generate-CTMR's diffusion from the same noise and differ
only in the BMI condition. The condition steers the whole denoising path, so they are not
the same individual at different weights; compare BMI columns across seeds. Each patient
runs through the same pipeline and quality metrics as ``generate_cohort.py``.

Example::

    python bmi_sweep.py --output sweep --bmi 20 25 30 35 40 --sex M F --age 55 --seeds 3 \\
        --source-root ~/NV-Generate-CTMR --python ~/NV-Generate-CTMR/.venv/bin/python --workers 3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_cohort import checkout_commit, process_patient  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    """Generate every (seed, sex, BMI) combination; returns 1 if any patient failed."""
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--bmi", type=float, nargs="+", default=[20, 25, 30, 35, 40])
    cli.add_argument("--sex", nargs="+", default=["M", "F"], choices=("M", "F"))
    cli.add_argument("--age", type=float, default=55)
    cli.add_argument("--seeds", type=int, default=3, help="Generation seeds 0..N-1 shared by every BMI")
    cli.add_argument("--source-root", type=Path, required=True)
    cli.add_argument("--python")
    cli.add_argument("--workers", type=int, default=1)
    cli.add_argument("--spacing", type=float, nargs=3, default=(1.9, 1.9, 2.5))
    args = cli.parse_args(argv)
    source_root = args.source_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in range(args.seeds):
        for sex in args.sex:
            for bmi in args.bmi:
                patient = {"id": f"seed{seed}_{sex}_bmi{bmi:g}", "sex": sex, "age": args.age, "bmi": bmi, "seed": seed}
                jobs.append({"folder": str(output / "patients" / patient["id"]), "patient": patient,
                             "demographics": {"age": args.age, "sex": sex, "bmi": bmi}, "source_root": str(source_root),
                             "python": args.python, "spacing": list(args.spacing), "output_size": [256, 256, 256],
                             "bundle_vessels": ["aorta"], "centerline_spacing_mm": 1.5})
    (output / "sweep.json").write_text(json.dumps({"bmi": args.bmi, "sex": args.sex, "age": args.age, "seeds": args.seeds,
                                                   "spacing_mm": list(args.spacing),
                                                   "nv_generate_commit": checkout_commit(source_root)}, indent=1))
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=multiprocessing.get_context("spawn")) as pool:
        records = list(pool.map(process_patient, jobs))
    for record in records:
        print(record["id"], record["status"], record.get("error", ""))
    return 0 if all(r["status"] == "ok" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
