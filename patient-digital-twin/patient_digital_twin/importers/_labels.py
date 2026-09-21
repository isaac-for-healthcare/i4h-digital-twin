# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private shared construction of anatomy from segmentation label names."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from ..catalog import CATALOG
from ..structures import AnatomicalStructure, Kind

if TYPE_CHECKING:
    from ..human import HumanBody


def body_from_labels(
    labels: Mapping[int, str] | Iterable[str],
    *,
    body: HumanBody | None = None,
    strict: bool = True,
    overrides: Mapping[str, Kind] | None = None,
) -> HumanBody:
    """Import an ID map or names (names get synthetic 1-based IDs).

    strict=True rejects unknown labels before changing the body. With False,
    they are retained as UNKNOWN. Background label 0 is ignored. Matching is
    exact and case-sensitive. Overrides extend or replace curated kinds.
    Repeated names share one structure; source label IDs are not retained.
    """
    from ..human import HumanBody

    if isinstance(labels, (str, bytes)):
        raise TypeError("Pass an iterable of names, not a single string")
    items = list(
        labels.items() if isinstance(labels, Mapping) else enumerate(labels, start=1)
    )
    catalog = {**CATALOG, **(overrides or {})}
    body = HumanBody() if body is None else body
    pending = []
    unknown = set()
    for label_id, name in items:
        if type(label_id) is not int or label_id < 0:
            raise ValueError(f"Invalid label ID: {label_id!r}")
        if label_id == 0:
            continue
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid label name: {name!r}")
        kind = Kind(catalog.get(name, Kind.UNKNOWN))
        if kind == Kind.UNKNOWN:
            unknown.add(name)
        existing = body.structures.get(name)
        if existing and existing.kind != kind:
            raise ValueError(f"Conflicting kind for {name}")
        pending.append((name, kind))
    if strict and unknown:
        raise ValueError("Unmapped labels: " + ", ".join(sorted(unknown)))
    for name, kind in pending:
        body.structures.setdefault(name, AnatomicalStructure(name, kind))
    body.anatomy.apply_configuration()
    return body
