# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private label construction and catalog validation tests."""

import pytest
from patient_digital_twin import (
    Kind,
)
from patient_digital_twin.importers._labels import anatomy_from_labels


def test_repeated_imports_share_structures_without_retaining_label_ids():
    body = anatomy_from_labels({0: "background", 1: "liver"})
    liver = body.structures["liver"]
    anatomy_from_labels({7: "liver"}, anatomy=body)
    assert body.structures["liver"] is liver
    assert not hasattr(liver, "labels")
    assert list(body.structures) == ["liver"]
    anatomy_from_labels({1: "heart"}, anatomy=body)
    assert list(body.structures) == ["liver", "heart"]


def test_unknown_labels_and_explicit_kind():
    body = anatomy_from_labels(["custom"], strict=False)
    assert body.structures["custom"].kind is Kind.UNKNOWN
    body = anatomy_from_labels(["custom"], overrides={"custom": Kind.ORGAN})
    assert body.structures["custom"].kind is Kind.ORGAN
    with pytest.raises(ValueError, match="Conflicting"):
        anatomy_from_labels(["custom"], anatomy=body, overrides={"custom": Kind.BONE})


@pytest.mark.parametrize("labels", ["liver", {-1: "liver"}, {True: "liver"}, {1: ""}])
def test_invalid_label_inputs(labels):
    with pytest.raises((TypeError, ValueError)):
        anatomy_from_labels(labels)
