# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private label construction and catalog validation tests."""

import pytest
from patient_digital_twin import (
    Kind,
)
from patient_digital_twin.importers._segmentation import anatomy_from_labels


def test_duplicate_names_share_one_structure():
    body = anatomy_from_labels(["liver", "liver", "heart"])
    assert list(body.structures) == ["liver", "heart"]
    assert body.structures["liver"].kind is Kind.ORGAN


def test_unknown_labels_require_non_strict_import():
    assert anatomy_from_labels(["custom"], strict=False).structures["custom"].kind is Kind.UNKNOWN
    with pytest.raises(ValueError, match="Unmapped"):
        anatomy_from_labels(["custom"])
