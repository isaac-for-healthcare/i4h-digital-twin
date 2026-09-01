# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclasses for CT preprocessing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HuToMuMapping:
    """Piecewise-linear Hounsfield Unit to linear attenuation coefficient mapping.

    The curve is defined by control points and is clamped outside the outermost pair,
    which is the same construction as window/level control on a radiology viewer. With
    the default two control points ``P0 = (hu_min, mu_min)`` and ``P1 = (hu_max, mu_max)``:

        mu(HU) = mu_min                                HU <= hu_min
        mu(HU) = mu_min + slope * (HU - hu_min)         hu_min < HU < hu_max
        mu(HU) = mu_max                                HU >= hu_max

        slope = (mu_max - mu_min) / (hu_max - hu_min)

    The two degrees of freedom that matter when tuning image appearance are the position
    of the ramp on the HU axis (level) and its steepness (window). ``from_window_level``,
    ``with_window_level``, ``shifted`` and ``scaled`` address those directly.

    Passing more than two ``control_points`` gives independent slopes per HU band (air,
    soft tissue, contrast, bone). That is supported, but every extra knot makes the curve
    harder to tune by hand, so prefer starting from the two-point ramp.

    Attributes:
        hu_min: HU value where attenuation starts to rise. Default: -1000 (air).
        hu_max: HU value where attenuation saturates. Default: 3000 (dense bone).
        mu_min: mu value (mm^-1) applied at and below hu_min. Default: 0.0.
        mu_max: mu value (mm^-1) applied at and above hu_max. Default: 0.02.
        control_points: Optional ``((HU, mu), ...)`` knots with strictly increasing HU.
            When given, these define the curve and the four scalar fields above are
            overwritten with the first and last knot so that they stay consistent.

    Suggested starting points for visual tuning. These are display-style settings to be
    adjusted against reference images, not calibrations against a kVp energy spectrum:

    | Emphasis                           | window_center | window_width |
    |------------------------------------|---------------|--------------|
    | Whole HU range (default)           | 1000          | 4000         |
    | Soft tissue and contrasted vessels | 100           | 800          |
    | Bone and dense structures          | 800           | 2000         |

    Example:
        >>> mapping = HuToMuMapping.from_window_level(window_center=100.0, window_width=800.0)
        >>> (mapping.hu_min, mapping.hu_max)
        (-300.0, 500.0)
        >>> mapping.scaled(1.5).mu_max  # steeper ramp, more contrast
        0.03
        >>> mapping.shifted(200.0).window_center  # slide the ramp along the HU axis
        300.0
    """

    hu_min: float = -1000.0
    hu_max: float = 3000.0
    mu_min: float = 0.0
    mu_max: float = 0.02
    control_points: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.control_points is None:
            self._validate_ramp(self.hu_min, self.hu_max, self.mu_min, self.mu_max)
            return

        points = tuple((float(hu), float(mu)) for hu, mu in self.control_points)
        if len(points) < 2:
            raise ValueError(f"control_points needs at least 2 knots, got {len(points)}")
        for (hu_lo, _), (hu_hi, _) in zip(points, points[1:]):
            if hu_hi <= hu_lo:
                raise ValueError(f"control_points must have strictly increasing HU, got {points}")
        for _, mu in points:
            if mu < 0.0:
                raise ValueError(f"control_points must have non-negative mu, got {points}")

        object.__setattr__(self, "control_points", points)
        object.__setattr__(self, "hu_min", points[0][0])
        object.__setattr__(self, "mu_min", points[0][1])
        object.__setattr__(self, "hu_max", points[-1][0])
        object.__setattr__(self, "mu_max", points[-1][1])

    @staticmethod
    def _validate_ramp(hu_min: float, hu_max: float, mu_min: float, mu_max: float) -> None:
        if hu_max <= hu_min:
            raise ValueError(f"hu_max must be greater than hu_min, got hu_min={hu_min}, hu_max={hu_max}")
        if mu_min < 0.0 or mu_max < 0.0:
            raise ValueError(f"mu values must be non-negative, got mu_min={mu_min}, mu_max={mu_max}")

    @property
    def points(self) -> tuple[tuple[float, float], ...]:
        """Return the control points ``((HU, mu), ...)`` defining the curve."""
        if self.control_points is not None:
            return self.control_points
        return ((self.hu_min, self.mu_min), (self.hu_max, self.mu_max))

    @property
    def hu_knots(self) -> tuple[float, ...]:
        """Return the HU coordinates of the control points, strictly increasing."""
        return tuple(hu for hu, _ in self.points)

    @property
    def mu_knots(self) -> tuple[float, ...]:
        """Return the mu coordinates (mm^-1) of the control points."""
        return tuple(mu for _, mu in self.points)

    @property
    def window_width(self) -> float:
        """Return the HU span covered by the ramp (hu_max - hu_min)."""
        return self.hu_max - self.hu_min

    @property
    def window_center(self) -> float:
        """Return the HU value at the middle of the ramp."""
        return 0.5 * (self.hu_min + self.hu_max)

    @property
    def slope(self) -> float:
        """Return the end-to-end gradient (mu per HU) across the ramp."""
        return (self.mu_max - self.mu_min) / self.window_width

    @classmethod
    def from_window_level(
        cls,
        window_center: float,
        window_width: float,
        mu_max: float = 0.02,
        mu_min: float = 0.0,
    ) -> "HuToMuMapping":
        """Build a two-point ramp from window/level parameters.

        Args:
            window_center: HU value at the middle of the ramp (level).
            window_width: HU span covered by the ramp (window). Must be positive.
            mu_max: mu value (mm^-1) at and above the top of the ramp.
            mu_min: mu value (mm^-1) at and below the bottom of the ramp.

        Returns:
            Mapping whose ramp spans ``[center - width/2, center + width/2]``.

        Raises:
            ValueError: If window_width is not positive.
        """
        if window_width <= 0.0:
            raise ValueError(f"window_width must be positive, got {window_width}")
        half = 0.5 * window_width
        return cls(
            hu_min=window_center - half,
            hu_max=window_center + half,
            mu_min=mu_min,
            mu_max=mu_max,
        )

    def with_window_level(
        self,
        window_center: float | None = None,
        window_width: float | None = None,
    ) -> "HuToMuMapping":
        """Return a mapping repositioned and rescaled on the HU axis.

        The mu values and the relative spacing of any intermediate control points are
        preserved; only the HU axis is remapped onto the requested window.

        Args:
            window_center: New level. Defaults to the current window_center.
            window_width: New window. Defaults to the current window_width.

        Returns:
            New mapping with the requested window/level.

        Raises:
            ValueError: If window_width is not positive.
        """
        center = self.window_center if window_center is None else window_center
        width = self.window_width if window_width is None else window_width
        if width <= 0.0:
            raise ValueError(f"window_width must be positive, got {width}")

        scale = width / self.window_width
        new_lo = center - 0.5 * width
        return self._rebuilt((new_lo + (hu - self.hu_min) * scale, mu) for hu, mu in self.points)

    def shifted(self, delta_hu: float) -> "HuToMuMapping":
        """Return a mapping translated along the HU axis (level control).

        Args:
            delta_hu: HU offset added to every control point.

        Returns:
            New mapping with the same shape at a different HU position.
        """
        return self._rebuilt((hu + delta_hu, mu) for hu, mu in self.points)

    def scaled(self, factor: float) -> "HuToMuMapping":
        """Return a mapping with all mu values scaled (contrast control).

        Args:
            factor: Non-negative multiplier applied to every mu control value.

        Returns:
            New mapping with a steeper (factor > 1) or flatter (factor < 1) ramp.

        Raises:
            ValueError: If factor is negative.
        """
        if factor < 0.0:
            raise ValueError(f"factor must be non-negative, got {factor}")
        return self._rebuilt((hu, mu * factor) for hu, mu in self.points)

    def _rebuilt(self, points: Iterable[tuple[float, float]]) -> "HuToMuMapping":
        knots = tuple(points)
        if len(knots) == 2:
            (hu_lo, mu_lo), (hu_hi, mu_hi) = knots
            return HuToMuMapping(hu_min=hu_lo, hu_max=hu_hi, mu_min=mu_lo, mu_max=mu_hi)
        return HuToMuMapping(control_points=knots)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON/YAML-friendly dictionary."""
        data: dict[str, Any] = {
            "hu_min": self.hu_min,
            "hu_max": self.hu_max,
            "mu_min": self.mu_min,
            "mu_max": self.mu_max,
        }
        if self.control_points is not None:
            data["control_points"] = [list(point) for point in self.control_points]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HuToMuMapping":
        """Create a mapping from a config dictionary.

        Accepts either explicit ``control_points``, a ``window_center``/``window_width``
        pair, or the ``hu_min``/``hu_max``/``mu_min``/``mu_max`` endpoints, in that
        order of precedence.

        Args:
            data: Config dictionary, e.g. parsed from YAML or read back from metadata.

        Returns:
            Mapping built from the recognised keys.
        """
        if data.get("control_points"):
            return cls(control_points=tuple((float(hu), float(mu)) for hu, mu in data["control_points"]))

        defaults = cls()
        mu_min = float(data.get("mu_min", defaults.mu_min))
        mu_max = float(data.get("mu_max", defaults.mu_max))

        if "window_center" in data and "window_width" in data:
            return cls.from_window_level(
                window_center=float(data["window_center"]),
                window_width=float(data["window_width"]),
                mu_min=mu_min,
                mu_max=mu_max,
            )

        return cls(
            hu_min=float(data.get("hu_min", defaults.hu_min)),
            hu_max=float(data.get("hu_max", defaults.hu_max)),
            mu_min=mu_min,
            mu_max=mu_max,
        )


@dataclass(frozen=True)
class PreprocessingSettings:
    """CT preprocessing settings."""

    hu_clip_min: float = -1024.0
    hu_clip_max: float = 3071.0
    clip_hu: bool = True
    hu_to_mu: HuToMuMapping = field(default_factory=HuToMuMapping)
