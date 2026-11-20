# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a virtual patient cohort with NV-Generate-CTMR and turn every patient into a digital twin.

For each patient sampled from a ``cohort.CohortSpec`` preset, this script:

1. generates a paired CT and label volume with ``NVGenerateImporter``, conditioned on the
   patient's characteristics (by default age, sex, and BMI);
2. runs the patient-digital-twin pipeline: meshes every labeled structure, attaches the CT,
   extracts each vessel's centerline, and exports an i4h-workflows bundle;
3. measures quality (``qc.py``): CT intensities, labels, meshes, centerlines, and the bundle.

Outputs, under ``--output``: ``cohort.json`` / ``cohort.csv`` (the sampled cohort and the run
settings) and ``patients/<id>/`` with ``ct.nii.gz``, ``labels.nii.gz``, ``twin/`` (the bundle),
``pipeline.log``, and ``record.json``. Rerunning the same command resumes; patients with a
``record.json`` are skipped. Summarize a cohort with ``make_report.py``.

Example::

    python generate_cohort.py --output cohort --count 500 --workers 3 \\
        --source-root ~/NV-Generate-CTMR --python ~/NV-Generate-CTMR/.venv/bin/python
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import shutil
import sys
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import qc  # noqa: E402
from patient_digital_twin import HumanBody, NVGenerateImporter  # noqa: E402
from patient_digital_twin.anatomy import is_vessel  # noqa: E402
from patient_digital_twin.importers import NV_GENERATE_COMMIT  # noqa: E402
from presets import PRESETS  # noqa: E402


def _write_json(path: Path, data: Any) -> None:
    """Write JSON atomically, so a record is either complete or absent."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=1))
    temporary.replace(path)


def _log(folder: Path, text: str) -> None:
    """Append to the patient's pipeline.log (independent of where sys.stderr points)."""
    with open(folder / "pipeline.log", "a") as log:
        log.write(text)


@contextmanager
def _output_to(path: Path) -> Iterator[None]:
    """Send this process's stdout/stderr, and its subprocesses', to a log file."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(path, "ab") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def ct_nifti(scan: Any) -> nib.Nifti1Image:
    """A millimeter NIfTI of a generated ``ScanVolume`` (HU, float32)."""
    affine = np.array(scan.ijk_to_ras_m, dtype=float)
    affine[:3] *= 1000
    image = nib.Nifti1Image(np.asarray(scan.values_kji, dtype=np.float32).transpose(2, 1, 0), affine)
    image.header.set_xyzt_units("mm")
    return image


def process_patient(job: Mapping[str, Any]) -> dict[str, Any]:
    """Generate, build, export, and measure one patient; never raises (failures are recorded)."""
    folder, patient = Path(job["folder"]), job["patient"]
    record_path = folder / "record.json"
    if record_path.is_file():
        return json.loads(record_path.read_text())
    if folder.exists():
        shutil.rmtree(folder)  # An interrupted attempt; this script owns the folder.
    folder.mkdir(parents=True)
    record: dict[str, Any] = {"id": patient["id"], "patient": patient, "demographics": job["demographics"],
                              "status": "failed", "timings_s": {}}
    timings, started = record["timings_s"], time.perf_counter()
    with _output_to(folder / "pipeline.log"):
        try:
            generator = NVGenerateImporter(
                source_root=job["source_root"], python_executable=job["python"], demographics=job["demographics"],
                seed=patient["seed"], spacing=job["spacing"], output_size=job["output_size"],
            )
            anatomy = generator.to_anatomy_collection()  # Every catalog structure present in the labels.
            timings["generate_and_mesh"] = round(time.perf_counter() - started, 1)
            nib.save(generator.label_image, folder / "labels.nii.gz")
            nib.save(ct_nifti(generator.ct_scan), folder / "ct.nii.gz")
            body = HumanBody(anatomy)
            body.attach_scan(generator.ct_scan)
            step, failures = time.perf_counter(), {}
            for name in [n for n, s in anatomy.structures.items() if not s.is_empty and is_vessel(n)]:
                try:  # One vessel at a time: extract_topology stores nothing if any vessel fails.
                    body.extract_topology(names=[name], spacing_m=job["centerline_spacing_mm"] / 1000)
                except Exception as exc:  # noqa: BLE001 - recorded as a QC finding.
                    failures[name] = f"{type(exc).__name__}: {exc}"
            timings["centerlines"] = round(time.perf_counter() - step, 1)
            step, manifest = time.perf_counter(), None
            vessels = [v for v in job["bundle_vessels"] if v in anatomy.structures and not anatomy.structures[v].is_empty]
            try:
                manifest = body.export_patient_twin(folder / "twin", patient_id=patient["id"], vessel_names=vessels,
                                                    ct_exterior=True)
            except Exception as exc:  # noqa: BLE001 - recorded as a QC finding.
                record["bundle_error"] = f"{type(exc).__name__}: {exc}"
                _log(folder, traceback.format_exc())
            timings["bundle"] = round(time.perf_counter() - step, 1)
            step = time.perf_counter()
            record["qc"] = qc.evaluate(ct_path=folder / "ct.nii.gz", labels_path=folder / "labels.nii.gz",
                                       anatomy=anatomy, manifest=manifest, source_root=job["source_root"])
            record["qc"]["centerline_failures"] = failures
            record["source"] = generator.ct_scan.metadata["source"]
            timings["qc"] = round(time.perf_counter() - step, 1)
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - one failed patient must not stop the cohort.
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            _log(folder, record["traceback"])
    timings["total"] = round(time.perf_counter() - started, 1)
    _write_json(record_path, record)
    return record


def checkout_commit(root: Path) -> str | None:
    """The git commit of a checkout, or None outside git."""
    import subprocess

    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() or None if result.returncode == 0 else None


def write_cohort(output: Path, settings: Mapping[str, Any], patients: Sequence[Mapping[str, Any]]) -> None:
    """Write ``cohort.json`` (settings and patients) and ``cohort.csv``, or check a resumed run matches."""
    path = output / "cohort.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing["settings"] != json.loads(json.dumps(settings)):
            raise SystemExit(f"{path} was written with different settings; use a new --output to start over")
        return
    output.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"settings": settings, "patients": list(patients)})
    with open(output / "cohort.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patients[0]))
        writer.writeheader()
        writer.writerows(patients)


def parser() -> argparse.ArgumentParser:
    """Command-line options; see the module docstring."""
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--output", type=Path, required=True, help="Cohort folder (created; reused to resume)")
    result.add_argument("--count", type=int, default=500, help="Number of patients (default 500)")
    result.add_argument("--seed", type=int, default=20261119, help="Cohort sampling seed")
    result.add_argument("--preset", default="ct_patients", help=f"Cohort specification: {', '.join(sorted(PRESETS))}")
    result.add_argument("--source-root", type=Path, default=os.environ.get("NV_GENERATE_ROOT"),
                        help="NV-Generate-CTMR checkout (default $NV_GENERATE_ROOT)")
    result.add_argument("--python", help="Python interpreter with NV-Generate-CTMR's requirements (separate process)")
    result.add_argument("--workers", type=int, default=1, help="Patients processed in parallel (about 10 GB RAM each)")
    result.add_argument("--spacing", type=float, nargs=3, default=(1.9, 1.9, 2.5), metavar=("X", "Y", "Z"),
                        help="Voxel size in mm (default 1.9 1.9 2.5: 486 x 486 x 640 mm, lung apices to pelvis)")
    result.add_argument("--output-size", type=int, nargs=3, default=(256, 256, 256), metavar=("X", "Y", "Z"))
    result.add_argument("--bundle-vessels", nargs="*", default=["aorta"],
                        help="Vessels rasterized with navigation centerlines in each bundle (default aorta)")
    result.add_argument("--centerline-spacing-mm", type=float, default=1.5)
    result.add_argument("--sample-only", action="store_true", help="Write cohort.json/csv without generating")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Sample the cohort, then process pending patients with a worker pool."""
    args = parser().parse_args(argv)
    if args.preset not in PRESETS:
        raise SystemExit(f"Unknown preset {args.preset!r}; choose from {sorted(PRESETS)}")
    spec = PRESETS[args.preset]()
    patients = spec.sample(args.count, args.seed)
    source_root = None if args.source_root is None else args.source_root.expanduser().resolve()
    settings = {
        "preset": args.preset, "count": args.count, "seed": args.seed, "conditioning": list(spec.conditioning),
        "spec": spec.describe(), "spacing_mm": list(args.spacing), "output_size": list(args.output_size),
        "bundle_vessels": list(args.bundle_vessels), "centerline_spacing_mm": args.centerline_spacing_mm,
        "source_root": None if source_root is None else str(source_root),
        "nv_generate_commit": None if source_root is None else checkout_commit(source_root),
    }
    output = args.output.expanduser().resolve()
    write_cohort(output, settings, patients)
    print(f"Cohort of {len(patients)} ({args.preset}) in {output / 'cohort.csv'}")
    if args.sample_only:
        return 0
    if source_root is None:
        raise SystemExit("Pass --source-root (an NV-Generate-CTMR checkout) or set NV_GENERATE_ROOT")
    if settings["nv_generate_commit"] not in (None, NV_GENERATE_COMMIT):
        print(f"warning: {source_root} is not at the tested commit {NV_GENERATE_COMMIT}", file=sys.stderr)
    jobs = [{"folder": str(output / "patients" / p["id"]), "patient": p, "demographics": spec.demographics(p),
             "source_root": str(source_root), "python": args.python, "spacing": list(args.spacing),
             "output_size": list(args.output_size), "bundle_vessels": list(args.bundle_vessels),
             "centerline_spacing_mm": args.centerline_spacing_mm} for p in patients]
    pending = [job for job in jobs if not (Path(job["folder"]) / "record.json").is_file()]
    print(f"{len(jobs) - len(pending)} done, {len(pending)} to process with {args.workers} worker(s)")
    started, failures = time.time(), 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=context) as pool:
        futures = [pool.submit(process_patient, job) for job in pending]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            failures += record["status"] != "ok"
            remaining = (time.time() - started) / done * (len(pending) - done)
            p = record["patient"]
            print(f"[{done}/{len(pending)}] {record['id']} {record['status']} {record['timings_s'].get('total')} s "
                  f"({p.get('sex')}, {p.get('age')} y, BMI {p.get('bmi')}) failures={failures} "
                  f"eta={time.strftime('%H:%M:%S', time.gmtime(remaining))}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
