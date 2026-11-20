# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cohort generation tutorial: distributions, presets, QC metrics, and the per-patient runner."""

import json
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

TUTORIAL = Path(__file__).resolve().parents[1] / "tutorials/cohort_generation"
sys.path.insert(0, str(TUTORIAL))

import cohort as cohort_api  # noqa: E402
import presets  # noqa: E402
import qc  # noqa: E402
from cohort import (  # noqa: E402
    Binomial,
    Categorical,
    CohortSpec,
    Conditional,
    Derived,
    Fixed,
    Histogram,
    LogNormal,
    Normal,
    Uniform,
    lognormal_from_percentiles,
    lognormal_from_prevalence,
    weight_from_bmi,
)


def draw(distribution, count=20000, seed=0, patient=None):
    rng = np.random.default_rng(seed)
    return [distribution.sample(rng, patient or {}) for _ in range(count)]


def test_binomial_counts_follow_binomial():
    values = draw(Binomial(p=0.3), count=4000)
    males = values.count("M")
    assert set(values) == {"M", "F"} and abs(males - 1200) < 4 * math.sqrt(4000 * 0.3 * 0.7)
    with pytest.raises(ValueError):
        Binomial(p=1.5)


@pytest.mark.parametrize(
    "distribution, low, high, mean",
    [
        (Normal(175, 7, 160, 190), 160, 190, 175),
        (Normal(60, 15, 19, 87), 19, 87, None),
        (LogNormal(28, 0.25, 18, 63), 18, 63, None),
        (Uniform(1, 3), 1, 3, 2),
        (Histogram((19, 30, 50, 87), (1, 2, 1)), 19, 87, None),
    ],
)
def test_truncated_draws_stay_in_range_and_match_density(distribution, low, high, mean):
    values = np.array(draw(distribution))
    assert values.min() >= low and values.max() <= high
    if mean is not None:
        assert abs(values.mean() - mean) < 0.2
    grid = np.linspace(low - 5, high + 5, 4001)
    density = distribution.pdf(grid)
    assert abs(np.trapezoid(density, grid) - 1) < 0.01  # Truncated densities are renormalized.
    counts, edges = np.histogram(values, bins=10, range=(low, high))
    expected = [np.trapezoid(distribution.pdf(x), x) for x in (np.linspace(a, b, 401) for a, b in zip(edges[:-1], edges[1:]))]
    assert np.max(np.abs(counts / len(values) - np.asarray(expected))) < 0.01  # Bin masses match the target.


def test_shifted_lognormal_matches_percentiles_and_nhanes_bmi():
    median, sigma, shift = lognormal_from_percentiles(20.8, 28.5, 41.0)  # NHANES 2015-2018 men.
    distribution = LogNormal(median, sigma, shift=shift)
    values = np.array(draw(distribution, count=200000))
    np.testing.assert_allclose(np.percentile(values, [5, 50, 95]), [20.8, 28.5, 41.0], atol=0.15)
    assert abs(values.mean() - 29.4) < 0.1  # The reported NHANES mean, not used in the fit.
    assert abs(distribution.share_above(30) - 0.403) < 0.002 and distribution.share_above(5) == 1.0
    truncated = LogNormal(median, sigma, 18, 63, shift=shift)
    grid = np.linspace(0, 80, 8001)
    assert abs(np.trapezoid(truncated.pdf(grid), grid) - 1) < 0.01
    with pytest.raises(ValueError, match="right-skewed"):
        lognormal_from_percentiles(20, 30, 35)


def test_lognormal_from_prevalence_recovers_shares():
    median, sigma = lognormal_from_prevalence(30, 0.403, 40, 0.094)
    values = np.array(draw(LogNormal(median, sigma), count=200000))
    assert abs((values >= 30).mean() - 0.403) < 0.005 and abs((values >= 40).mean() - 0.094) < 0.005
    with pytest.raises(ValueError):
        lognormal_from_prevalence(30, 0.1, 40, 0.2)


def test_spec_conditional_derived_and_reproducible():
    spec = CohortSpec({
        "sex": Binomial(p=0.5),
        "height": Conditional("sex", {"M": Fixed(180.0), "F": Fixed(160.0)}),
        "bmi": Categorical((20.0, 30.0), (1, 1)),
        "weight": Derived(weight_from_bmi, "BMI x height^2"),
        "age": Uniform(19, 87),
    })
    patients = spec.sample(50, seed=3)
    assert patients == spec.sample(50, seed=3) and patients != spec.sample(50, seed=4)
    for patient in patients:
        assert patient["height"] == {"M": 180.0, "F": 160.0}[patient["sex"]]
        assert patient["weight"] == round(patient["bmi"] * (patient["height"] / 100) ** 2, 1)
        assert 0 <= patient["seed"] < 2**32
    assert spec.demographics(patients[0]) == {k: patients[0][k] for k in ("age", "sex", "bmi")}
    assert [p["id"] for p in patients[:2]] == ["P0000", "P0001"]
    assert [row["characteristic"] for row in spec.describe()] == ["sex", "height", "bmi", "weight", "age"]


@pytest.mark.parametrize(
    "characteristics, conditioning, message",
    [
        ({"age": Fixed(50.0)}, ("age", "sex"), "not sampled"),
        ({"height": Fixed(170.0)}, ("height",), "cannot condition"),
        ({"bmi": Fixed(25.0), "weight": Fixed(70.0)}, ("bmi", "weight"), "not both"),
        ({"seed": Fixed(1.0)}, (), "reserved"),
    ],
)
def test_spec_rejects_unsupported_conditioning(characteristics, conditioning, message):
    with pytest.raises(ValueError, match=message):
        CohortSpec(characteristics, conditioning=conditioning)


@pytest.mark.parametrize("name", sorted(presets.PRESETS))
def test_presets_sample_within_the_model_training_ranges(name):
    spec = presets.PRESETS[name]()
    patients = spec.sample(300, seed=1)
    for key, (low, high) in cohort_api.TRAINING_RANGE.items():
        if key in spec.conditioning:
            assert all(low <= p[key] <= high for p in patients)
    assert set(spec.conditioning) <= set(cohort_api.CONDITIONABLE)
    assert all(row["source"] for row in spec.describe()), "every preset characteristic cites a source"


def _write_upstream(root):
    (root / "configs").mkdir(parents=True)
    (root / "configs/label_dict.json").write_text(json.dumps({
        "liver": 1, "aorta": 6, "vertebrae L3": 35, "left lung upper lobe": 28, "body": 200}))
    statistics = {"liver": {"sigma_6_low": 0, "sigma_6_high": 150, "percentile_0_5": 10, "percentile_99_5": 140}}
    (root / "configs/image_median_statistics_ct.json").write_text(json.dumps(statistics))


def test_qc_label_and_ct_metrics(tmp_path):
    _write_upstream(tmp_path)
    ids = qc.label_ids(tmp_path)
    assert ids["liver"] == [1] and ids["lung_upper_lobe_left"] == [28]
    labels = np.zeros((20, 20, 20), np.int32)
    labels[2:18, 2:18, :] = 200  # Body, cut by the superior and inferior faces.
    labels[4:8, 4:8, 4:8] = 1  # Liver, plus a detached 1-voxel island.
    labels[12, 12, 12] = 1
    labels[9:11, 9:11, 0:20] = 6  # Aorta touching the inferior and superior faces.
    labels[13:15, 13:15, 9:12] = 35  # L3 around slice 10.
    metrics = qc.label_metrics(labels, np.array([2.0, 2.0, 2.0]), ids)
    assert metrics["volume_ml"]["liver"] == pytest.approx(65 * 8 / 1000)
    assert metrics["core_touching_boundary"]["aorta"] == ["inferior", "superior"]
    assert metrics["fragments"]["liver"] == {"components": 2, "outside_largest": pytest.approx(1 / 65, abs=1e-5)}
    assert set(metrics["body_touching_boundary"]) == {"inferior", "superior"}
    assert "aorta" not in metrics["core_missing"] and "heart" in metrics["core_missing"]
    ct = np.full(labels.shape, -1000.0, np.float32)
    ct[labels > 0] = -100  # Adipose.
    ct[labels == 1] = 60
    statistics = json.loads((tmp_path / "configs/image_median_statistics_ct.json").read_text())
    result = qc.ct_metrics(ct, labels, np.array([2.0, 2.0, 2.0]), ids, statistics)
    assert result["hu"]["liver"] == {"median": 60.0, "low": 0.0, "high": 150.0, "outlier": False}
    assert result["l3"]["slice"] == 10 and result["l3"]["body_area_cm2"] == pytest.approx(16 * 16 * 4 / 100, abs=0.05)
    assert result["l3"]["adipose_area_cm2"] == pytest.approx(16 * 16 * 4 / 100, abs=0.05)  # Every body voxel is -100 HU.
    assert result["air_median_hu"] == -1000.0


def test_qc_mesh_metrics_detect_open_and_nonmanifold_meshes():
    from patient_digital_twin import AnatomyCollection
    from patient_digital_twin.geometry import mask_to_mesh

    cube = np.zeros((6, 6, 6), np.uint8)
    cube[1:5, 1:5, 1:5] = 1
    vertices, faces = mask_to_mesh(cube)
    tetrahedron = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    # Two closed tetrahedra glued along the edge 0-1: that edge has four triangles.
    glued = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1]], float) * 0.01
    anatomy = AnatomyCollection.from_names(["liver", "spleen", "aorta"])
    for name, (v, f) in {"liver": (vertices * 0.001, faces), "spleen": (vertices * 0.001, faces[1:]),
                         "aorta": (glued, np.vstack([tetrahedron, [[0, 4, 1], [0, 1, 5], [1, 4, 5], [0, 5, 4]]]))}.items():
        anatomy.structures[name].vertices, anatomy.structures[name].faces = v, f
    metrics = qc.mesh_metrics(anatomy, {"liver": 0.064, "aorta": 0.001})
    liver, spleen, aorta = (metrics["structures"][n] for n in ("liver", "spleen", "aorta"))
    assert liver["boundary_edges"] == liver["nonmanifold_edges"] == 0 and liver["components"] == 1
    assert spleen["boundary_edges"] == 3 and spleen["nonmanifold_edges"] == 0
    assert aorta["boundary_edges"] == 0 and aorta["nonmanifold_edges"] == 1 and aorta["components"] == 1
    assert metrics["closed_manifold"] == 1 and metrics["not_closed_manifold"] == ["aorta", "spleen"]
    assert metrics["structures"]["liver"]["volume_ml"] == pytest.approx(64 * 1e-3, rel=0.2)  # 4 mm cube, cut corners.
    assert metrics["core_volume_error"]["liver"] == pytest.approx(liver["volume_ml"] / 0.064 - 1, abs=1e-3)


def test_process_patient_records_failures_and_resumes(monkeypatch, tmp_path):
    pytest.importorskip("vtk")
    import generate_cohort

    calls = []

    class Failing:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def to_anatomy_collection(self):
            raise RuntimeError("no GPU")

    monkeypatch.setattr(generate_cohort, "NVGenerateImporter", Failing)
    patient = {"id": "P0000", "sex": "F", "age": 50.0, "bmi": 30.0, "seed": 5}
    job = {"folder": str(tmp_path / "P0000"), "patient": patient, "demographics": {"age": 50.0, "sex": "F", "bmi": 30.0},
           "source_root": str(tmp_path), "python": None, "spacing": [1.9, 1.9, 2.5], "output_size": [256, 256, 256],
           "bundle_vessels": ["aorta"], "centerline_spacing_mm": 1.5}
    record = generate_cohort.process_patient(job)
    assert record["status"] == "failed" and record["error"] == "RuntimeError: no GPU"
    assert calls == [{"source_root": str(tmp_path), "python_executable": None, "demographics": job["demographics"],
                      "seed": 5, "spacing": [1.9, 1.9, 2.5], "output_size": [256, 256, 256]}]
    assert "no GPU" in (tmp_path / "P0000/pipeline.log").read_text()
    assert generate_cohort.process_patient(job) == record and len(calls) == 1  # Resumed from record.json.


def test_ct_nifti_round_trip(tmp_path):
    import generate_cohort
    from patient_digital_twin.scan_volume import from_nifti

    affine = np.diag([1.9, 1.9, 2.5, 1.0])
    affine[:3, 3] = [-10, 5, 30]
    source = nib.Nifti1Image(np.arange(60, dtype=np.float32).reshape(3, 4, 5), affine)
    source.header.set_xyzt_units("mm")
    nib.save(source, tmp_path / "ct.nii.gz")
    image = generate_cohort.ct_nifti(from_nifti(tmp_path / "ct.nii.gz"))
    np.testing.assert_allclose(image.affine, affine)
    np.testing.assert_array_equal(np.asanyarray(image.dataobj), np.asanyarray(source.dataobj))


def test_report_cancels_back_to_back_pairs_and_judges_records():
    pytest.importorskip("matplotlib")
    import make_report
    from patient_digital_twin.geometry import mask_to_mesh

    cube = np.zeros((5, 5, 5), np.uint8)
    cube[1:4, 1:4, 1:4] = 1
    vertices, faces = mask_to_mesh(cube)
    existing = {tuple(sorted(face)) for face in faces.tolist()}
    extra = next(t for t in ((0, i, j) for i in range(1, len(vertices)) for j in range(i + 1, len(vertices)))
                 if t not in existing)
    doubled = np.vstack([faces, [extra, extra[::-1]]])  # A back-to-back pair, as marching cubes can emit.
    assert (qc.edge_counts(doubled, len(vertices)) > 2).any()
    kept = make_report.cancel_back_to_back(doubled)
    np.testing.assert_array_equal(kept, faces)
    assert (qc.edge_counts(kept, len(vertices)) == 2).all()

    def record(**qc_overrides):
        metrics = {"ct": {"hu": {"liver": {"outlier": False}}},
                   "labels": {"core_missing": [], "core_touching_boundary": {}, "body_touching_boundary": ["inferior"],
                              "fragments": {"liver": {"outside_largest": 0.0}, make_report.PORTAL: {"outside_largest": 0.3}}},
                   "mesh": {"not_closed_manifold": ["rib_left_3"]}, "centerlines": {"aorta": {"components": 1}},
                   "bundle": {"usd_prims_found": 5, "structures": 5, "vessel_mask_matches_volume": True, "centerline_points": 9}}
        metrics.update(qc_overrides)
        return {"status": "ok", "qc": metrics}

    good = make_report.verdicts(record())
    assert good["twin_ready"] and good["body_in_fov"] and not good["portal_vein_connected"]
    assert not make_report.verdicts(record(mesh={"not_closed_manifold": ["liver"]}))["twin_ready"]
    assert not make_report.verdicts(record(centerlines={"aorta": {"components": 2}}))["aorta_centerline"]
    failed = make_report.verdicts({"status": "failed"})
    assert not any(failed.values())
