# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for how the preprocess-ct CLI builds the HU to mu transfer function."""

from __future__ import annotations

import numpy as np
import pytest
from vasculature_digital_twin import HuToMuMapping, hu_to_mu
from vasculature_digital_twin.cli.preprocess_ct import (
    _build_mapping,
    _parse_control_points,
    build_parser,
)


def _mapping_from(*argv: str) -> HuToMuMapping:
    """Build the mapping the CLI would use for the given HU-to-mu arguments."""
    args = build_parser().parse_args(["--nifti", "ct.nii.gz", "--output-dir", "out", *argv])
    return _build_mapping(args)


class TestControlPoints:
    """--control-points reaches the multi-knot curve that the window/level flags cannot express."""

    def test_knots_are_parsed_in_order(self):
        mapping = _mapping_from("--control-points=-1000:0,0:0.004,300:0.012,1500:0.02")
        assert mapping.points == ((-1000.0, 0.0), (0.0, 0.004), (300.0, 0.012), (1500.0, 0.02))

    def test_curve_has_independent_slopes_per_band(self):
        """The point of the flag: one slope per band rather than a single clipped ramp."""
        mapping = _mapping_from("--control-points=-1000:0,0:0.004,300:0.012,1500:0.02")
        mu = hu_to_mu(np.array([-500.0, 150.0, 900.0]), mapping)
        np.testing.assert_allclose(mu, [0.002, 0.008, 0.016], rtol=1e-6)

    def test_endpoints_are_exposed_as_the_ramp_bounds(self):
        mapping = _mapping_from("--control-points=-1000:0,0:0.004,1500:0.02")
        assert (mapping.hu_min, mapping.mu_min) == (-1000.0, 0.0)
        assert (mapping.hu_max, mapping.mu_max) == (1500.0, 0.02)

    def test_two_knots_are_accepted(self):
        mapping = _mapping_from("--control-points=-300:0,500:0.02")
        assert mapping.points == ((-300.0, 0.0), (500.0, 0.02))

    def test_surrounding_whitespace_is_tolerated(self):
        mapping = _mapping_from("--control-points= -1000:0 , 0:0.004 , 1500:0.02 ")
        assert mapping.points == ((-1000.0, 0.0), (0.0, 0.004), (1500.0, 0.02))

    @pytest.mark.parametrize(
        "value",
        ["-1000:0,0:0.004,1500:0.02", "0:0,1000:0.02", " -1000:0,1500:0.02"],
    )
    def test_parser_accepts_realistic_values(self, value):
        assert len(_parse_control_points(value)) >= 2


class TestControlPointsRejection:
    """Bad curves fail at the CLI boundary with a message naming the flag."""

    def test_missing_colon_rejected(self):
        with pytest.raises(SystemExit, match="expects hu:mu pairs"):
            _mapping_from("--control-points=-1000:0,300")

    def test_non_numeric_rejected(self):
        with pytest.raises(SystemExit, match="expects numeric hu:mu pairs"):
            _mapping_from("--control-points=-1000:0,soft:0.01")

    def test_single_knot_rejected(self):
        with pytest.raises(SystemExit, match="at least 2 knots"):
            _mapping_from("--control-points=0:0.01")

    def test_non_increasing_hu_rejected(self):
        with pytest.raises(SystemExit, match="strictly increasing HU"):
            _mapping_from("--control-points=1500:0.02,0:0.004")

    def test_negative_mu_rejected(self):
        with pytest.raises(SystemExit, match="non-negative mu"):
            _mapping_from("--control-points=0:0,1000:-0.01")

    @pytest.mark.parametrize(
        "conflict",
        [
            ["--window-center", "100", "--window-width", "800"],
            ["--mu-max", "0.03"],
        ],
    )
    def test_conflicting_ramp_flags_rejected(self, conflict):
        with pytest.raises(SystemExit, match="cannot be combined with"):
            _mapping_from("--control-points=-1000:0,1500:0.02", *conflict)

    def test_space_separated_negative_value_is_rejected_by_argparse(self):
        """Documents why the help text insists on --control-points=... for negative HU."""
        with pytest.raises(SystemExit):
            _mapping_from("--control-points", "-1000:0,1500:0.02")


class TestWindowLevelUnchanged:
    """The existing two-point paths keep behaving exactly as before."""

    def test_default_is_the_standard_ramp(self):
        assert _mapping_from() == HuToMuMapping()

    def test_window_level_builds_a_two_knot_ramp(self):
        mapping = _mapping_from("--window-center", "100", "--window-width", "800")
        assert mapping.points == ((-300.0, 0.0), (500.0, 0.02))

    def test_mu_max_still_applies_to_the_default_ramp(self):
        assert _mapping_from("--mu-max", "0.05").mu_max == 0.05

    def test_mu_max_still_applies_to_the_window_level_ramp(self):
        mapping = _mapping_from("--window-center", "100", "--window-width", "800", "--mu-max", "0.05")
        assert mapping.points == ((-300.0, 0.0), (500.0, 0.05))

    @pytest.mark.parametrize("partial", [["--window-center", "100"], ["--window-width", "800"]])
    def test_window_flags_must_be_paired(self, partial):
        with pytest.raises(SystemExit, match="must be given together"):
            _mapping_from(*partial)
