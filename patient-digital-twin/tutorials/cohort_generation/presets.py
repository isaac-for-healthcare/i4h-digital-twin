# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ready-made cohort specifications with literature sources (README.md, "Choosing distributions").

- ``ct_patients``: adults undergoing CT (sex and age of 2.0 million CT examinations), with
  US adult height and BMI by sex. Used for the 500-patient report.
- ``us_adults``: the US adult population (2020 Census sex and age, NHANES height and BMI).
- ``nv_generate_training``: the demographics of NV-Generate-CTMR's own training cases.

Ages and BMIs are restricted to NV-Generate-CTMR's training ranges (``cohort.TRAINING_RANGE``).
Add a preset by writing a function that returns a ``CohortSpec`` and listing it in ``PRESETS``.
"""

from __future__ import annotations

from collections.abc import Callable

from cohort import (
    TRAINING_RANGE,
    Binomial,
    CohortSpec,
    Conditional,
    Derived,
    Histogram,
    LogNormal,
    Normal,
    lognormal_from_percentiles,
    weight_from_bmi,
)

AGE_LOW, AGE_HIGH = TRAINING_RANGE["age"]
BMI_LOW, BMI_HIGH = TRAINING_RANGE["bmi"]
Z_95 = 1.6449  # Standard normal 95th percentile.

SMITH_BINDMAN_2019 = ("Smith-Bindman et al., BMJ 2019;364:k4931, Table 1: 2,005,459 adult CT examinations at "
                      "151 institutions, 2015-2017")
CENSUS_2020 = "2020 US Census, DHC table P12, adults aged 20+"
FRYAR_2021 = "Fryar et al., NHANES 2015-2018 anthropometric reference data, Vital Health Stat 3(46), 2021"
ZHAO_2026 = "Zhao et al., NV-Generate-Body, arXiv:2610.24153 (2026), Table 5"

#: CT examinations by patient age (18-29, ..., 70-79, 80+) and the female count (Smith-Bindman 2019).
CT_EXAM_AGE_EDGES = (AGE_LOW, 30, 40, 50, 60, 70, 80, AGE_HIGH)
CT_EXAM_AGE_COUNTS = (179_889, 177_964, 251_366, 385_556, 439_115, 337_904, 233_665)
CT_EXAM_FEMALE, CT_EXAM_TOTAL = 1_056_929, 2_005_459
#: US adults aged 20+ (2020 Census): males, total, and each sex's share by age band (20-29, ..., 80+).
CENSUS_MALES_20, CENSUS_ADULTS_20 = 120_701_572, 249_254_717
CENSUS_AGE_EDGES = (20, 30, 40, 50, 60, 70, 80, AGE_HIGH)
CENSUS_AGE_SHARE = {"M": (18.64, 18.31, 16.53, 17.35, 15.62, 9.41, 4.15),
                    "F": (17.09, 17.35, 15.95, 17.04, 16.12, 10.44, 6.00)}
#: NHANES 2015-2018 adults 20+ (Fryar 2021, Tables 9, 11, 14, 15): height mean and 5th/95th
#: percentiles (cm), BMI 5th/50th/95th percentiles (kg/m²).
NHANES_HEIGHT = {"M": (175.3, 162.8, 187.4), "F": (161.3, 149.8, 172.5)}
NHANES_BMI_PERCENTILES = {"M": (20.8, 28.5, 41.0), "F": (19.7, 28.4, 44.6)}


def height(sex: str) -> Normal:
    """Height (cm): normal with the NHANES mean, SD from the 5th-95th percentile spread, truncated at ±3 SD."""
    mean, p5, p95 = NHANES_HEIGHT[sex]
    sd = round((p95 - p5) / (2 * Z_95), 2)
    return Normal(mean, sd, round(mean - 3 * sd, 1), round(mean + 3 * sd, 1))


def bmi(sex: str) -> LogNormal:
    """BMI (kg/m²): three-parameter log-normal through the NHANES 5th/50th/95th percentiles, in the model's range."""
    median, sigma, shift = lognormal_from_percentiles(*NHANES_BMI_PERCENTILES[sex])
    return LogNormal(median, round(sigma, 4), BMI_LOW, BMI_HIGH, shift=round(shift, 3))


def _anthropometrics() -> dict[str, Conditional | Derived]:
    return {
        "height": Conditional("sex", {"M": height("M"), "F": height("F")}),
        "bmi": Conditional("sex", {"M": bmi("M"), "F": bmi("F")}),
        "weight": Derived(weight_from_bmi, "BMI x (height / 100)^2"),
    }


_ANTHROPOMETRIC_SOURCES = {"height": FRYAR_2021, "bmi": FRYAR_2021, "weight": "derived from BMI and height"}


def ct_patients() -> CohortSpec:
    """Adults undergoing CT: sex and age of CT examinations; US adult height and BMI by sex."""
    return CohortSpec(
        {
            "sex": Binomial(p=round(1 - CT_EXAM_FEMALE / CT_EXAM_TOTAL, 4)),
            "age": Histogram(CT_EXAM_AGE_EDGES, CT_EXAM_AGE_COUNTS),
            **_anthropometrics(),
        },
        conditioning=("age", "sex", "bmi"),
        name="ct_patients",
        sources={"sex": SMITH_BINDMAN_2019, "age": SMITH_BINDMAN_2019, **_ANTHROPOMETRIC_SOURCES},
    )


def us_adults() -> CohortSpec:
    """US adults aged 20-87: 2020 Census sex and age by sex; NHANES height and BMI by sex."""
    return CohortSpec(
        {
            "sex": Binomial(p=round(CENSUS_MALES_20 / CENSUS_ADULTS_20, 4)),
            "age": Conditional("sex", {sex: Histogram(CENSUS_AGE_EDGES, CENSUS_AGE_SHARE[sex]) for sex in ("M", "F")}),
            **_anthropometrics(),
        },
        conditioning=("age", "sex", "bmi"),
        name="us_adults",
        sources={"sex": CENSUS_2020, "age": CENSUS_2020, **_ANTHROPOMETRIC_SOURCES},
    )


def nv_generate_training() -> CohortSpec:
    """NV-Generate-CTMR's training demographics (mean and SD only, so normal distributions)."""
    return CohortSpec(
        {
            "sex": Binomial(p=0.602),
            "age": Normal(64.1, 12.0, AGE_LOW, AGE_HIGH),
            "bmi": Normal(29.3, 6.3, BMI_LOW, BMI_HIGH),
        },
        conditioning=("age", "sex", "bmi"),
        name="nv_generate_training",
        sources={key: ZHAO_2026 for key in ("sex", "age", "bmi")},
    )


PRESETS: dict[str, Callable[[], CohortSpec]] = {
    "ct_patients": ct_patients, "us_adults": us_adults, "nv_generate_training": nv_generate_training,
}
