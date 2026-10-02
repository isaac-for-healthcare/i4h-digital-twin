# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HumanBody.extract_topology selection, local-frame storage, and atomic commits."""

import numpy as np
import pytest
from patient_digital_twin import AnatomicalStructure, CenterlineGraph, HumanBody, Kind
from patient_digital_twin import body as body_module
from patient_digital_twin.body import CATALOG


def graph():
    return CenterlineGraph(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
        np.array([0.01, 0.01]),
        np.array([[0, 1]]),
    )


def test_body_extracts_tubular_anatomy_including_disabled_and_composite(monkeypatch):
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    names = {
        "aorta": Kind.VESSEL,
        "trachea": CATALOG["trachea"][0],
        "bronchus": Kind.AIRWAY,
        "portal_vein_and_splenic_vein": Kind.GROUP,
        "liver": Kind.ORGAN,
        "lung_left": Kind.ORGAN,
    }
    body = HumanBody(
        {
            n: AnatomicalStructure(n, k, vertices.copy(), np.array([[0, 1, 2]]))
            for n, k in names.items()
        }
    )
    body.anatomy.structures["aorta"].enabled = False
    body.anatomy.structures["empty"] = AnatomicalStructure("empty", Kind.VESSEL)
    calls = []

    def extract(v, f, **grid):
        calls.append((v, f))
        return graph()

    monkeypatch.setattr(body_module, "extract_centerlines", extract)
    result = body.extract_topology()
    assert set(result) == {
        "aorta",
        "trachea",
        "bronchus",
        "portal_vein_and_splenic_vein",
    }
    assert len(calls) == 4
    structure = body.anatomy.structures["aorta"]
    assert structure.centerline is result["aorta"]
    structure.local_to_world[:3, 3] = 1
    np.testing.assert_array_equal(structure.centerline.points, graph().points)
    structure.enabled = True
    assert structure.centerline is result["aorta"]
    structure.vertices = vertices.copy()
    assert structure.centerline is None
    structure.centerline = graph()
    structure.faces = np.array([[0, 2, 1]])
    assert structure.centerline is None


def test_failure_is_named_and_does_not_commit_partial_results(monkeypatch):
    body = HumanBody(
        {
            n: AnatomicalStructure(
                n,
                Kind.VESSEL,
                np.ones((3, 3)),
                np.array([[0, 1, 2]]),
                centerline=graph(),
            )
            for n in ["first", "second"]
        }
    )
    previous = body.anatomy.structures["first"].centerline

    def extract(v, f, **grid):
        if v is body.anatomy.structures["second"].mesh.vertices:
            raise ValueError("bad tube")
        return graph()

    monkeypatch.setattr(body_module, "extract_centerlines", extract)
    with pytest.raises(RuntimeError, match="second: bad tube"):
        body.extract_topology()
    assert body.anatomy.structures["first"].centerline is previous


def test_empty_body_and_invalid_spacing():
    assert HumanBody().extract_topology() == {}
    with pytest.raises(ValueError, match="spacing_m"):
        HumanBody().extract_topology(spacing_m=0)


def test_topology_selection_preserves_other_centerlines(monkeypatch):
    body = HumanBody(
        {
            n: AnatomicalStructure(
                n,
                Kind.VESSEL,
                np.ones((3, 3)),
                np.array([[0, 1, 2]]),
                centerline=graph(),
            )
            for n in ["aorta", "vascular_tree"]
        }
    )
    original = body.anatomy.structures["aorta"].centerline
    monkeypatch.setattr(body_module, "extract_centerlines", lambda v, f, **grid: graph())
    assert set(body.extract_topology(names=["vascular_tree"])) == {"vascular_tree"}
    assert body.anatomy.structures["aorta"].centerline is original
    with pytest.raises(KeyError, match="Unknown anatomy"):
        body.extract_topology(names=["missing"])
