# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility artifacts preserve physical grids and supplied centerline graphs."""

import json

import numpy as np
import pytest

from patient_digital_twin.legacy_ct import write_artifacts
from patient_digital_twin.legacy_ct.ct.dicom_ingest import CtVolume


def volume():
    z, y, x = np.indices((30, 15, 15))
    mask = ((x - 7)**2 + (y - 7)**2 < 16) & (z > 2) & (z < 27)
    ct = CtVolume(np.where(mask, 300, -1000).astype(np.float32),
                  (2., 1., 1.), (20., 30., 40.), tuple(np.eye(3).ravel()), 'LPS', 'SPL')
    return ct, mask


def test_navigation_artifacts_have_physical_centerlines(tmp_path):
    ct, mask = volume()
    write_artifacts(ct, tmp_path, vessel_mask=mask)
    points = np.load(tmp_path/'centerline_points_mm.npy')
    indices = np.rint((points-ct.origin_xyz_mm)/np.asarray(ct.spacing_zyx_mm)[::-1]).astype(int)[:, ::-1]
    assert mask[tuple(indices.T)].all()
    edges = np.load(tmp_path/'centerline_edges.npy')
    assert len(edges) and edges.max() < len(points)
    assert np.all(np.load(tmp_path/'centerline_radii_mm.npy') > 0)
    metadata = json.loads((tmp_path/'metadata.json').read_text())
    assert metadata['anatomical_frame'] == 'LPS'
    assert metadata['hu_to_mu']['preset'] == 'interventional'
    np.testing.assert_array_equal(np.load(tmp_path/'vessel_mask.npy'), mask)
    np.testing.assert_array_equal(np.load(tmp_path/'hu_volume.npy'), ct.hu_zyx)


def test_supplied_graph_is_preserved(tmp_path, monkeypatch):
    import patient_digital_twin.legacy_ct.artifacts as module
    ct, mask = volume()
    graph = (np.array([[27.,37.,50.],[27.,37.,52.]]), np.array([[0,1]]), np.array([3.,3.]))
    monkeypatch.setattr(module, 'centerline_from_mask', lambda *a: pytest.fail('Existing graph must be reused'))
    write_artifacts(ct, tmp_path, vessel_mask=mask, centerline=graph)
    np.testing.assert_array_equal(np.load(tmp_path/'centerline_points_mm.npy'), graph[0])


def test_invalid_mask_leaves_no_artifacts(tmp_path):
    ct, _ = volume()
    with pytest.raises(ValueError, match='Vessel mask'):
        write_artifacts(ct, tmp_path, vessel_mask=np.ones((2,2,2)))
    assert not list(tmp_path.iterdir())
