# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result types exposed by :mod:`imaging_to_mesh`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeshArtifact:
    """One anatomical mesh produced by a conversion."""

    name: str
    obj_path: Path
    vertex_count: int
    face_count: int


@dataclass(frozen=True)
class ConversionResult:
    """Files and meshes produced by a conversion."""

    usd_path: Path
    meshes: tuple[MeshArtifact, ...]
    source_path: Path | None = None
