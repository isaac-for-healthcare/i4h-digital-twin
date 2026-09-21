# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from patient_digital_twin import HumanBody, Kind, System
from patient_digital_twin.geometry import rotation_between, solve_similarity
from patient_digital_twin.importers._labels import body_from_labels


def test_human_body_owns_shared_components_and_landmarks():
    from patient_digital_twin import AnatomyCollection, SomaRepresentation

    landmarks = {"left_shoulder": np.array([1.0, 2.0, 3.0])}
    body = HumanBody(source_path="scan.nii", landmarks=landmarks)
    assert isinstance(body.anatomy, AnatomyCollection)
    assert isinstance(body.soma, SomaRepresentation)
    assert body.soma.anatomy is body.anatomy
    assert body.structures is body.soma.structures
    assert body.landmarks is body.soma.landmarks is landmarks
    assert body.source_path == "scan.nii"
    assert body.soma_body is body.soma_layer is None
    assert (
        body.alignment_errors_m == body.shape_fit_report == body.bone_fit_report == {}
    )


@pytest.mark.parametrize(
    "method,args,kwargs",
    [
        ("attach_soma", ("layer",), {"device": "cpu"}),
        ("pose", ("pose-array",), {"transl": [[1, 2, 3]]}),
        ("check_containment", (), {"tolerance_m": 0.001}),
        ("update_from_soma", ({"vertices": "output"},), {}),
        ("fit_soma_shape", (["pose-array"],), {"iterations": 1}),
        ("fit_bone_anchors", (["pose-array"],), {"max_translation_m": 0.02}),
    ],
)
def test_human_api_delegates_model_work(method, args, kwargs):
    from unittest.mock import Mock

    from patient_digital_twin import SomaRepresentation

    body = HumanBody()
    body.soma = Mock(spec=SomaRepresentation)
    actual = getattr(body, method)(*args, **kwargs)
    target = getattr(body.soma, method)
    target.assert_called_once_with(*args, **kwargs)
    if method == "update_from_soma":
        assert actual is None
    else:
        assert actual is target.return_value


def test_human_body_and_catalog():
    body = body_from_labels({1: "liver", 2: "kidney_left", 3: "femur_left"})
    assert isinstance(body, HumanBody)
    assert [s.name for s in body.select(kind=Kind.BONE)] == ["femur_left"]
    assert [s.name for s in body.select(system=System.URINARY)] == ["kidney_left"]
    assert body.structures["liver"].vertices is None
    assert body.soma_layer is None


def test_strict_label_import_is_atomic():
    body = body_from_labels(["liver"])
    with pytest.raises(ValueError, match="Unmapped"):
        body_from_labels({2: "heart", 3: "unmapped"}, body=body)
    assert list(body.structures) == ["liver"]


def test_similarity_and_degenerate_landmarks():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    target = 2 * source @ rotation.T + [3, 4, 5]
    fit = solve_similarity(source, target)
    np.testing.assert_allclose(fit.apply(source), target, atol=1e-10)
    assert fit.scale == pytest.approx(2)
    assert np.linalg.det(fit.rotation) == pytest.approx(1)
    with pytest.raises(ValueError, match="non-collinear"):
        solve_similarity(np.zeros((3, 3)), np.zeros((3, 3)))
    assert np.linalg.norm(rotation_between([1, 0, 0], [-1, 0, 0])) == pytest.approx(
        np.pi
    )


def test_optional_imaging_transform_recovers_scan_coordinates_without_aliasing():
    from patient_digital_twin import AnatomicalStructure
    from patient_digital_twin.geometry import transform_points

    placement = np.eye(4)
    placement[:3, 3] = [0.2, 0.1, -0.3]
    structure = AnatomicalStructure(
        "liver",
        Kind.ORGAN,
        vertices=np.array([[0.0, 0, 0], [0.1, 0.2, 0.3]]),
        local_to_body=placement,
    )
    body = HumanBody({"liver": structure})
    assert body.body_to_imaging is None
    with pytest.raises(ValueError, match="body_to_imaging"):
        body.imaging_vertices("liver")
    imaging = np.array([[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1.0]])
    body.body_to_imaging = imaging
    expected = transform_points(structure.body_vertices, imaging)
    imaging[:3, 3] = 99
    copy = body.body_to_imaging
    copy[:3, 3] = 100
    structure.local_to_world[:3, 3] = [-1, -2, -3]
    np.testing.assert_allclose(body.imaging_vertices("liver"), expected)
    body.set_anatomy_enabled(False)
    assert body.imaging_vertices("liver") is None
    with pytest.raises(ValueError, match="rigid"):
        body.body_to_imaging = np.diag([2, 1, 1, 1])
    body.body_to_imaging = None
    assert body.body_to_imaging is None
