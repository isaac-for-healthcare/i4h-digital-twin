# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclasses for CT preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HuToMuMapping:
    """Hounsfield Unit to linear attenuation coefficient mapping."""

    hu_min: float = -1000.0
    hu_max: float = 3000.0
    mu_min: float = 0.0
    mu_max: float = 0.02


@dataclass(frozen=True)
class PreprocessingSettings:
    """CT preprocessing settings."""

    hu_clip_min: float = -1024.0
    hu_clip_max: float = 3071.0
    clip_hu: bool = True
    hu_to_mu: HuToMuMapping = field(default_factory=HuToMuMapping)
