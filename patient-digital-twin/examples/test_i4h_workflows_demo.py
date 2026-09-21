# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Similarity is symmetric and independent of graph sampling density."""

import numpy as np
import pytest
from i4h_workflows_demo import compare_graphs
from patient_digital_twin import CenterlineGraph


def test_similarity_accepts_different_sampling_and_rejects_shift_and_missing_branch():
    reference = CenterlineGraph(
        np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, 20.0, 0.0]]),
        np.ones(3),
        np.array([[0, 1], [1, 2]]),
    )
    actual = CenterlineGraph(
        np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, 20.0, 0.0]]
        )
        * 0.001,
        np.ones(4) * 0.001,
        np.array([[0, 1], [1, 2], [2, 3]]),
    )
    assert compare_graphs(reference, actual)["similar"]
    actual.points[:, 2] += 0.01
    assert not compare_graphs(reference, actual)["similar"]
    actual.points[:, 2] -= 0.01
    actual.edges = actual.edges[:2]
    report = compare_graphs(reference, actual)
    assert not report["similar"]
    assert report["reference_coverage"] < 0.95
    assert report["human_body_coverage"] == 1.0
    actual.edges = np.empty((0, 2), dtype=int)
    with pytest.raises(ValueError, match="without edges"):
        compare_graphs(reference, actual)
