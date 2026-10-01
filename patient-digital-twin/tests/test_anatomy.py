# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct collection and system-view tests."""

import numpy as np
import pytest
from patient_digital_twin import (
    AnatomicalStructure,
    AnatomicalSystem,
    AnatomyCollection,
    Kind,
    System,
)


def test_collection_filters_and_live_shared_system_views():
    liver = AnatomicalStructure(
        "liver",
        Kind.ORGAN,
        vertices=np.zeros((3, 3)),
    )
    members = {"liver": liver}
    anatomy = AnatomyCollection(members)
    view = anatomy.system("digestive")
    assert isinstance(view, AnatomicalSystem)
    assert view.anatomy is anatomy and anatomy.structures is members
    assert anatomy.select(kind=Kind.BONE) == []
    assert anatomy.select(region="thorax") == []
    assert anatomy.select(kind=Kind.ORGAN, region="abdomen", include_empty=False) == [
        liver
    ]
    assert view.structures == [liver] and not view.is_empty
    pancreas = AnatomicalStructure(
        "pancreas",
        Kind.ORGAN,
    )
    members["pancreas"] = pancreas
    assert view.structures[-1] is pancreas  # Existing views see newly added labels.
    assert anatomy.system(System.ENDOCRINE).structures[0] is pancreas
    view.set_enabled(False)
    assert view.is_empty and anatomy.select(include_empty=False) == []
    anatomy.set_structure_enabled("liver", True)
    assert not view.is_empty
    anatomy.set_enabled(False)
    assert view.is_empty
    anatomy.set_enabled(True)
    assert not view.is_empty


def test_empty_system_and_configuration_validation():
    anatomy = AnatomyCollection()
    assert anatomy.system("skeletal").is_empty
    assert anatomy.system("skeletal").structures == []
    with pytest.raises(ValueError):
        anatomy.system("misspelled")
    with pytest.raises(ValueError, match="Unknown"):
        anatomy.configure({"anatomy": {"structures": {"absent": False}}})
    assert anatomy.configuration.enabled
