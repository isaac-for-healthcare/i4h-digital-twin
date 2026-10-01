# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native artifacts and topology use a single physical mask-to-graph path."""

import numpy as np
import pytest
from patient_digital_twin.artifacts import write_artifacts, write_centerline
from patient_digital_twin.scan_volume import from_array, load_artifact
from patient_digital_twin.topology import native_centerline


def test_rotated_native_mask_artifacts(tmp_path):
    z, y, x = np.indices((30, 15, 15))
    mask = ((x-7)**2+(y-7)**2 < 16) & (z>2) & (z<27)
    affine = np.array([[0., -1., 0., 20.], [1., 0., 0., 30.], [0., 0., 2., 40.], [0., 0., 0., 1.]])
    scan = from_array(np.where(mask, 9000, -1500).transpose(2, 1, 0), affine)
    output = write_artifacts(scan, tmp_path/'out', vessel_mask=mask.transpose(2, 1, 0))
    loaded = load_artifact(output/'volume.yaml')
    np.testing.assert_array_equal(loaded.values, scan.values)
    points = np.load(output/'centerline_points.npy')
    indices = (np.c_[points, np.ones(len(points))] @ np.linalg.inv(affine).T)[:, :3]
    np.testing.assert_allclose(indices, np.rint(indices), atol=1e-6)
    assert mask[tuple(np.rint(indices[:, ::-1]).astype(int).T)].all()
    assert np.all(np.load(output/'centerline_radii.npy')>0)
    assert not (output/'mu_volume.npy').exists()
    with pytest.raises(FileExistsError):
        write_artifacts(scan, output)


def test_graph_writer_preserves_arrays_and_validates_before_writing(tmp_path):
    graph = (np.array([[1., 2., 3.], [1., 2., 4.]]), np.array([[0, 1]]), np.array([.1, .2]))
    paths = write_centerline(tmp_path/'graph', *graph)
    for name, data in zip(paths.values(), graph):
        np.testing.assert_array_equal(np.load(tmp_path/'graph'/name), data)
    with pytest.raises(ValueError):
        write_centerline(tmp_path/'invalid', graph[0], np.array([[0, 99]]), graph[2])
    assert not (tmp_path/'invalid').exists()


def test_bad_mask_fails_without_partial_export(tmp_path):
    scan = from_array(np.ones((3, 4, 5)), np.eye(4))
    for mask in [np.ones((2, 2, 2)), np.full(scan.values.shape, 2)]:
        with pytest.raises(ValueError):
            write_artifacts(scan, tmp_path/'bad', vessel_mask=mask)
    assert not (tmp_path/'bad').exists()
    affine = np.eye(4)
    affine[0, 1] = .2
    with pytest.raises(ValueError, match='orthogonal'):
        native_centerline(np.ones(scan.values.shape), from_array(scan.values, affine))
