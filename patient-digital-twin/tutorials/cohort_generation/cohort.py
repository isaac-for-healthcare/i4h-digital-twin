# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample virtual patient cohorts from literature-based distributions.

A ``CohortSpec`` lists named patient characteristics, each drawn from a distribution:

- ``Binomial``: two outcomes such as sex. One Bernoulli trial per patient, so the number
  of males in a cohort of n patients is Binomial(n, p).
- ``Normal`` / ``LogNormal`` / ``Uniform``: continuous values, optionally truncated to a
  range by exact inverse-CDF sampling (no clipping spikes at the bounds).
- ``Histogram`` (piecewise uniform, e.g. census age bands), ``Categorical``, ``Fixed``.
- ``Conditional``: a distribution chosen by an earlier characteristic, e.g. sex-specific
  height. ``Derived``: a value computed from earlier ones, e.g. weight from BMI and height.

Characteristics are sampled in order, so a characteristic can depend on earlier ones.
``CohortSpec.conditioning`` names the characteristics that NV-Generate-CTMR is conditioned
on: any of age, sex, and bmi or weight (see ``NVGenerateImporter``). Other characteristics,
such as height, describe the virtual patient but do not change the generated anatomy.

Example::

    spec = CohortSpec({
        "sex": Binomial(p=0.5),
        "age": Normal(mean=60, sd=12, low=19, high=87),
        "bmi": Conditional("sex", {"M": LogNormal(median=27, sigma=0.2, low=18, high=63),
                                   "F": LogNormal(median=26, sigma=0.25, low=18, high=63)}),
    })
    patients = spec.sample(500, seed=1)  # [{"id": "P0000", "sex": "F", "age": 57.3, ...}, ...]

``presets.py`` holds ready-made, literature-based specifications.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

Value = float | str
Patient = dict[str, Any]

#: Characteristics NV-Generate-CTMR's mask diffusion model is conditioned on.
CONDITIONABLE = ("age", "sex", "weight", "bmi")
#: Ranges of the NV-Generate-CTMR mask model's training demographics (upstream
#: ``scripts/sample_mask.py``); upstream warns that values outside them extrapolate.
TRAINING_RANGE = {"age": (19.0, 87.0), "bmi": (18.0, 63.0), "weight": (22.0, 144.0)}


class Distribution(ABC):
    """A characteristic's distribution: ``sample`` draws one patient's value."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, patient: Mapping[str, Any]) -> Value:
        """Draw a value; ``patient`` holds the characteristics sampled so far."""

    @abstractmethod
    def describe(self) -> str:
        """One-line description for reports, e.g. ``Normal(mean 175.3, SD 7.6) in [152.5, 198.1]``."""

    def pdf(self, x: np.ndarray, value: Value | None = None) -> np.ndarray | None:
        """Target density at ``x`` (or, for ``Conditional``, of case ``value``); None if not numeric."""
        return None


def _bounds(low: float, high: float) -> str:
    """`` in [low, high]`` for finite bounds, else an empty string."""
    if math.isinf(low) and math.isinf(high):
        return ""
    return f" in [{low:g}, {high:g}]"


def _truncated(dist: NormalDist, low: float, high: float, rng: np.random.Generator) -> float:
    """Exact draw from ``dist`` restricted to ``[low, high]`` by inverse-CDF sampling."""
    lo = dist.cdf(low) if math.isfinite(low) else 0.0
    hi = dist.cdf(high) if math.isfinite(high) else 1.0
    if not hi > lo:
        raise ValueError(f"The interval [{low}, {high}] has no probability mass")
    u = min(max(lo + (hi - lo) * float(rng.random()), 1e-12), 1 - 1e-12)
    return dist.inv_cdf(u)


def _check_interval(low: float, high: float) -> None:
    if not low < high:
        raise ValueError(f"Expected low < high, got [{low}, {high}]")


@dataclass(frozen=True)
class Binomial(Distribution):
    """Two outcomes with ``P(outcomes[0]) = p``: one Bernoulli trial per patient.

    A cohort of n patients therefore has Binomial(n, p) patients with the first outcome.
    """

    p: float
    outcomes: tuple[str, str] = ("M", "F")

    def __post_init__(self) -> None:
        if not 0 <= self.p <= 1 or len(set(self.outcomes)) != 2:
            raise ValueError("Binomial needs 0 <= p <= 1 and two distinct outcomes")

    def sample(self, rng: np.random.Generator, patient: Mapping[str, Any]) -> Value:
        return self.outcomes[0] if rng.random() < self.p else self.outcomes[1]

    def describe(self) -> str:
        return f"Binomial: P({self.outcomes[0]}) = {self.p:g}"


@dataclass(frozen=True)
class Normal(Distribution):
    """Gaussian ``N(mean, sd^2)``, optionally truncated to ``[low, high]``."""

    mean: float
    sd: float
    low: float = -math.inf
    high: float = math.inf

    def __post_init__(self) -> None:
        if not self.sd > 0:
            raise ValueError("Normal needs sd > 0")
        _check_interval(self.low, self.high)

    def sample(self, rng: np.random.Generator, patient: Mapping[str, Any]) -> Value:
        return _truncated(NormalDist(self.mean, self.sd), self.low, self.high, rng)

    def describe(self) -> str:
        return f"Normal(mean {self.mean:g}, SD {self.sd:g}){_bounds(self.low, self.high)}"

    def pdf(self, x: np.ndarray, value: Value | None = None) -> np.ndarray:
        dist = NormalDist(self.mean, self.sd)
        mass = (dist.cdf(self.high) if math.isfinite(self.high) else 1.0) - (
            dist.cdf(self.low) if math.isfinite(self.low) else 0.0)
        density = np.array([dist.pdf(float(v)) for v in np.ravel(x)]).reshape(np.shape(x)) / mass
        return np.where((x >= self.low) & (x <= self.high), density, 0.0)


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Right-skewed ``shift + exp(N(ln(median - shift), sigma^2))``, optionally truncated.

    ``sigma`` is the standard deviation of ln(value - shift). With ``shift = 0`` this is the
    usual log-normal; a positive ``shift`` (a three-parameter log-normal) lets the lower and
    upper tails differ, which fits adult BMI (see ``lognormal_from_percentiles``).
    """

    median: float
    sigma: float
    low: float = -math.inf
    high: float = math.inf
    shift: float = 0.0

    def __post_init__(self) -> None:
        if not self.median > self.shift or not self.sigma > 0:
            raise ValueError("LogNormal needs median > shift and sigma > 0")
        _check_interval(self.low, self.high)

    def _log_bounds(self) -> tuple[float, float]:
        return (math.log(self.low - self.shift) if self.low > self.shift else -math.inf,
                math.log(self.high - self.shift) if math.isfinite(self.high) else math.inf)

    def _log_normal(self) -> NormalDist:
        return NormalDist(math.log(self.median - self.shift), self.sigma)

    def sample(self, rng: np.random.Generator, patient: Mapping[str, Any]) -> Value:
        return self.shift + math.exp(_truncated(self._log_normal(), *self._log_bounds(), rng))

    def describe(self) -> str:
        shift = f", shift {self.shift:g}" if self.shift else ""
        return f"LogNormal(median {self.median:g}, log-SD {self.sigma:g}{shift}){_bounds(self.low, self.high)}"

    def pdf(self, x: np.ndarray, value: Value | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        excess = np.where(x > self.shift, x - self.shift, np.nan)
        inner = Normal(self._log_normal().mean, self.sigma, *self._log_bounds()).pdf(np.log(excess))
        density = np.nan_to_num(inner / excess)
        return np.where((x >= self.low) & (x <= self.high), density, 0.0)

    def share_above(self, threshold: float) -> float:
        """Untruncated probability of a value at or above ``threshold`` (e.g. obesity prevalence)."""
        return 1 - self._log_normal().cdf(math.log(threshold - self.shift)) if threshold > self.shift else 1.0


@dataclass(frozen=True)
class Uniform(Distribution):
    """Uniform on ``[low, high]``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        _check_interval(self.low, self.high)

    def sample(self, rng: np.random.Generator, patient: Mapping[str, Any]) -> Value:
        return float(rng.uniform(self.low, self.high))

    def describe(self) -> str:
        return f"Uniform[{self.low:g}, {self.high:g}]"

    def pdf(self, x: np.ndarray, value: Value | None = None) -> np.ndarray:
        return np.where((np.asarray(x) >= self.low) & (np.asarray(x) <= self.high), 1 / (self.high - self.low), 0.0)


@dataclass(frozen=True)
class Histogram(Distribution):
    """Piecewise-uniform density: bin ``edges`` with relative ``weights`` (e.g. census age bands)."""

    edges: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        edges, weights = np.asarray(self.edges, float), np.asarray(self.weights, float)
        if len(edges) != len(weights) + 1 or np.any(np.diff(edges) <= 0) or np.any(weights < 0) or not weights.sum() > 0:
            raise ValueError("Histogram needs increasing edges, one more edge than weights, and positive total weight")

    def sample(self, rng: np.random.Generator, patient: Mapping[str, Any]) -> Value:
        weights = np.asarray(self.weights, float)
        index = int(rng.choice(len(weights), p=weights / weights.sum()))
        return float(rng.uniform(self.edges[index], self.edges[index + 1]))

    def describe(self) -> str:
        return f"Histogram: {len(self.weights)} bins over [{self.edges[0]:g}, {self.edges[-1]:g})"

    def pdf(self, x: np.ndarray, value: Value | None = None) -> np.ndarray:
        edges, weights = np.asarray(self.edges, float), np.asarray(self.weights, float)
        density = weights / weights.sum() / np.diff(edges)
        index = np.searchsorted(edges, x, side="right") - 1
        inside = (index >= 0) & (index < len(weights))
        return np.where(inside, density[np.clip(index, 0, len(weights) - 1)], 0.0)


@dataclass(frozen=True)
class Categorical(Distribution):
    """One of ``values`` with relative ``weights``."""

    values: tuple[Value, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.weights) or not self.values or min(self.weights) < 0 or not sum(self.weights) > 0:
            raise ValueError("Categorical needs matching values and non-negative weights")

    def sample(self, rng: np.random.Generator, patient: Mapping[str, Any]) -> Value:
        weights = np.asarray(self.weights, float)
        return self.values[int(rng.choice(len(weights), p=weights / weights.sum()))]

    def describe(self) -> str:
        total = sum(self.weights)
        return "Categorical: " + ", ".join(f"{v} {w / total:.0%}" for v, w in zip(self.values, self.weights))


@dataclass(frozen=True)
class Fixed(Distribution):
    """The same value for every patient."""

    value: Value

    def sample(self, rng: np.random.Generator, patient: Mapping[str, Any]) -> Value:
        return self.value

    def describe(self) -> str:
        return f"Fixed: {self.value}"


@dataclass(frozen=True)
class Conditional(Distribution):
    """Choose a distribution by an earlier characteristic: ``Conditional("sex", {"M": ..., "F": ...})``."""

    on: str
    cases: Mapping[Value, Distribution]

    def sample(self, rng: np.random.Generator, patient: Mapping[str, Any]) -> Value:
        if self.on not in patient:
            raise KeyError(f"Sample {self.on!r} before characteristics that depend on it")
        return self.cases[patient[self.on]].sample(rng, patient)

    def describe(self) -> str:
        return "; ".join(f"{key}: {case.describe()}" for key, case in self.cases.items())

    def pdf(self, x: np.ndarray, value: Value | None = None) -> np.ndarray | None:
        return None if value is None else self.cases[value].pdf(x)


@dataclass(frozen=True)
class Derived(Distribution):
    """A value computed from earlier characteristics, e.g. ``weight = bmi * (height / 100) ** 2``."""

    function: Callable[[Mapping[str, Any]], Value]
    description: str

    def sample(self, rng: np.random.Generator, patient: Mapping[str, Any]) -> Value:
        return self.function(patient)

    def describe(self) -> str:
        return f"Derived: {self.description}"


@dataclass
class CohortSpec:
    """Named characteristics (sampled in order) and the subset that conditions NV-Generate.

    Attributes:
        characteristics: ``{name: Distribution}``; later entries may depend on earlier ones.
        conditioning: Characteristics passed to ``NVGenerateImporter(demographics=...)``.
        name / sources: Labels and literature sources (``{characteristic: citation}``) for reports.
        decimals: Rounding of sampled numbers (also applied before ``Derived`` values use them).
    """

    characteristics: Mapping[str, Distribution]
    conditioning: tuple[str, ...] = ("age", "sex", "bmi")
    name: str = "custom"
    sources: Mapping[str, str] = field(default_factory=dict)
    decimals: int = 1

    def __post_init__(self) -> None:
        self.conditioning = tuple(self.conditioning)
        if missing := set(self.conditioning) - set(self.characteristics):
            raise ValueError(f"Conditioning characteristics are not sampled: {sorted(missing)}")
        if unsupported := set(self.conditioning) - set(CONDITIONABLE):
            raise ValueError(f"NV-Generate cannot condition on {sorted(unsupported)}; use {list(CONDITIONABLE)}")
        if {"bmi", "weight"} <= set(self.conditioning):
            raise ValueError("Condition on bmi or weight, not both")
        if "id" in self.characteristics or "seed" in self.characteristics:
            raise ValueError("'id' and 'seed' are reserved patient fields")

    def sample(self, count: int, seed: int) -> list[Patient]:
        """Draw ``count`` patients reproducibly; each gets an ``id`` and its own generation ``seed``."""
        if count < 0:
            raise ValueError("count must be non-negative")
        rng = np.random.default_rng(seed)
        patients = []
        for index in range(count):
            patient: Patient = {"id": f"P{index:04d}"}
            for name, distribution in self.characteristics.items():
                value = distribution.sample(rng, patient)
                patient[name] = value if isinstance(value, str) else round(float(value), self.decimals)
            patient["seed"] = int(rng.integers(0, 2**32))  # NV-Generate accepts seeds in [0, 2**32).
            patients.append(patient)
        return patients

    def demographics(self, patient: Mapping[str, Any]) -> dict[str, Value]:
        """The patient's NV-Generate conditioning, e.g. ``{"age": 61.2, "sex": "F", "bmi": 29.4}``."""
        return {name: patient[name] for name in self.conditioning}

    def describe(self) -> list[dict[str, str]]:
        """One row per characteristic: name, distribution, whether it conditions NV-Generate, source."""
        return [{"characteristic": name, "distribution": distribution.describe(),
                 "conditions_nv_generate": "yes" if name in self.conditioning else "no",
                 "source": self.sources.get(name, "")}
                for name, distribution in self.characteristics.items()]


def weight_from_bmi(patient: Mapping[str, Any]) -> float:
    """Body weight (kg) = BMI (kg/m²) x height (m)²."""
    return float(patient["bmi"]) * (float(patient["height"]) / 100) ** 2


def bmi_from_weight(patient: Mapping[str, Any]) -> float:
    """BMI (kg/m²) = weight (kg) / height (m)²."""
    return float(patient["weight"]) / (float(patient["height"]) / 100) ** 2


def lognormal_from_percentiles(low: float, median: float, high: float, p: float = 0.05) -> tuple[float, float, float]:
    """``(median, sigma, shift)`` of the three-parameter log-normal with three given percentiles.

    ``low`` and ``high`` are the ``p`` and ``1 - p`` percentiles; they must be right-skewed
    around ``median``. Example: NHANES 2015-2018 men's BMI percentiles 20.8 (5th),
    28.5 (50th), and 41.0 (95th) give ``LogNormal(28.5, 0.295, shift=8.45)``.
    """
    spread = high + low - 2 * median
    if not low < median < high or not spread > 0:
        raise ValueError("Percentiles must increase and be right-skewed (high - median > median - low)")
    shift = (high * low - median**2) / spread
    sigma = math.log((high - shift) / (median - shift)) / NormalDist().inv_cdf(1 - p)
    return median, sigma, shift


def lognormal_from_prevalence(threshold_1: float, above_1: float, threshold_2: float, above_2: float) -> tuple[float, float]:
    """``(median, sigma)`` of the log-normal with the given fractions above two thresholds.

    Example: NHANES 2021-2023 adults had 40.3% with BMI >= 30 and 9.4% with BMI >= 40, so
    ``lognormal_from_prevalence(30, 0.403, 40, 0.094)`` gives a median BMI near 28.
    """
    z_1, z_2 = (NormalDist().inv_cdf(1 - p) for p in (above_1, above_2))
    sigma = (math.log(threshold_2) - math.log(threshold_1)) / (z_2 - z_1)
    if not sigma > 0:
        raise ValueError("The fraction above the higher threshold must be smaller")
    return math.exp(math.log(threshold_1) - z_1 * sigma), sigma


def table(rows: Sequence[Mapping[str, str]]) -> str:
    """A Markdown table from ``CohortSpec.describe()`` rows."""
    keys = list(rows[0]) if rows else []
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(row[k]) for k in keys) + " |" for row in rows]
    return "\n".join(lines)
