# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for vasculature-digital-twin tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_ct_hu() -> np.ndarray:
    """Build a small synthetic contrast CT with a tubular vessel along +Z."""
    shape = (40, 32, 32)
    hu = np.full(shape, -900.0, dtype=np.float32)
    # Bright tubular lumen through the volume center.
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cy, cx = shape[1] // 2, shape[2] // 2
    radius = 3.0
    tube = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
    # Keep a connected segment with enough voxels for min-component filtering.
    tube = tube & (zz >= 4) & (zz < shape[0] - 4)
    hu[tube] = 400.0
    return hu


@pytest.fixture
def spacing_zyx_mm() -> tuple[float, float, float]:
    return (1.0, 0.8, 0.8)
