# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DICOM ingest against a synthesised series with known geometry and HU values.

The series is written with ``ImageOrientationPatient`` describing rows running toward the
patient's Right and columns toward Anterior: a valid layout that needs two flips to reach
the canonical frame, so both the HU scaling and the reorientation are exercised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from vasculature_digital_twin import CANONICAL_FRAME

ROWS, COLUMNS, SLICES = 12, 10, 6
ROW_SPACING_MM, COLUMN_SPACING_MM = 0.8, 0.9
SLICE_THICKNESS_MM = 2.0
FIRST_SLICE_ORIGIN_MM = np.array([60.0, -40.0, -15.0])

# Rows run toward patient Right (-Left), columns toward Anterior (-Posterior).
ROW_DIRECTION = np.array([-1.0, 0.0, 0.0])
COLUMN_DIRECTION = np.array([0.0, -1.0, 0.0])
SLICE_DIRECTION = np.cross(ROW_DIRECTION, COLUMN_DIRECTION)

RESCALE_INTERCEPT = -1024.0
BACKGROUND_HU = -824.0
EDGE_HU = -924.0
FIDUCIAL_HU = 2500.0
FIDUCIAL_INDEX = (2, 5, 7)  # (slice, row, column) as stored


def _fiducial_position_mm() -> np.ndarray:
    """LPS position of the fiducial, derived only from the DICOM geometry tags."""
    slice_index, row, column = FIDUCIAL_INDEX
    return (
        FIRST_SLICE_ORIGIN_MM
        + slice_index * SLICE_THICKNESS_MM * SLICE_DIRECTION
        + row * ROW_SPACING_MM * COLUMN_DIRECTION
        + column * COLUMN_SPACING_MM * ROW_DIRECTION
    )


def _brightest_index(volume: np.ndarray) -> tuple[int, int, int]:
    """Locate the fiducial by rank, keeping geometry checks independent of HU scaling."""
    return tuple(int(i) for i in np.unravel_index(int(np.argmax(volume)), volume.shape))


def _darkest_columns(slice_yx: np.ndarray) -> list[int]:
    """Columns holding the edge marker, located by rank rather than absolute HU."""
    return sorted(set(int(c) for c in np.argwhere(np.isclose(slice_yx, slice_yx.min()))[:, 1]))


def _voxel_position_mm(
    index_zyx: tuple[int, int, int],
    origin_xyz_mm: tuple[float, float, float],
    direction: tuple[float, ...],
    spacing_zyx_mm: tuple[float, float, float],
) -> np.ndarray:
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    position = np.asarray(origin_xyz_mm, dtype=np.float64)
    for axis, index in enumerate(index_zyx):
        position = position + index * spacing_zyx_mm[axis] * matrix[:, 2 - axis]
    return position


@pytest.fixture
def dicom_series(tmp_path: Path) -> Path:
    """Write a small CT series and return its directory."""
    pydicom = pytest.importorskip("pydicom")
    pytest.importorskip("SimpleITK")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    directory = tmp_path / "series"
    directory.mkdir()
    series_uid = generate_uid()
    study_uid = generate_uid()
    frame_of_reference_uid = generate_uid()

    for slice_index in range(SLICES):
        stored = np.full((ROWS, COLUMNS), BACKGROUND_HU - RESCALE_INTERCEPT, dtype=np.uint16)
        stored[:, :2] = EDGE_HU - RESCALE_INTERCEPT  # asymmetric edge, detects a column flip
        if slice_index == FIDUCIAL_INDEX[0]:
            stored[FIDUCIAL_INDEX[1], FIDUCIAL_INDEX[2]] = int(FIDUCIAL_HU - RESCALE_INTERCEPT)

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.Modality = "CT"
        ds.PatientID = "ORIENT001"
        ds.InstanceNumber = slice_index + 1

        ds.Rows = ROWS
        ds.Columns = COLUMNS
        ds.PixelSpacing = [ROW_SPACING_MM, COLUMN_SPACING_MM]
        ds.SliceThickness = SLICE_THICKNESS_MM
        ds.ImageOrientationPatient = [*ROW_DIRECTION, *COLUMN_DIRECTION]
        ds.ImagePositionPatient = list(FIRST_SLICE_ORIGIN_MM + slice_index * SLICE_THICKNESS_MM * SLICE_DIRECTION)
        ds.RescaleIntercept = RESCALE_INTERCEPT
        ds.RescaleSlope = 1.0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = stored.tobytes()

        pydicom.dcmwrite(directory / f"slice_{slice_index:03d}.dcm", ds, enforce_file_format=True)

    return directory


class TestHounsfieldScaling:
    """The modality LUT must be applied exactly once."""

    def test_hu_values_match_the_tags(self, dicom_series: Path):
        from vasculature_digital_twin.ct.dicom_ingest import load_dicom_series_hu

        hu = load_dicom_series_hu(dicom_series).hu_zyx

        # Applying Rescale Intercept twice would shift all three values by -1024 HU.
        np.testing.assert_allclose(sorted(np.unique(hu)), [EDGE_HU, BACKGROUND_HU, FIDUCIAL_HU])

    def test_air_and_soft_tissue_stay_in_range(self, dicom_series: Path):
        from vasculature_digital_twin.ct.dicom_ingest import load_dicom_series_hu

        hu = load_dicom_series_hu(dicom_series).hu_zyx
        assert float(hu.min()) >= -1024.0


class TestDicomOrientation:
    """Direction cosines from the tags drive the reorientation."""

    def test_source_orientation_is_read_from_the_tags(self, dicom_series: Path):
        from vasculature_digital_twin.ct.dicom_ingest import load_dicom_series_hu

        ct = load_dicom_series_hu(dicom_series)

        assert ct.source_orientation == "SAR"
        assert ct.anatomical_frame == CANONICAL_FRAME

    def test_direction_is_canonical_after_reorientation(self, dicom_series: Path):
        from vasculature_digital_twin.ct.dicom_ingest import load_dicom_series_hu

        ct = load_dicom_series_hu(dicom_series)
        np.testing.assert_allclose(np.asarray(ct.direction).reshape(3, 3), np.eye(3), atol=1e-9)

    def test_spacing_follows_the_permuted_axes(self, dicom_series: Path):
        from vasculature_digital_twin.ct.dicom_ingest import load_dicom_series_hu

        ct = load_dicom_series_hu(dicom_series)
        assert ct.spacing_zyx_mm == pytest.approx((SLICE_THICKNESS_MM, ROW_SPACING_MM, COLUMN_SPACING_MM))

    def test_fiducial_keeps_its_patient_space_position(self, dicom_series: Path):
        from vasculature_digital_twin.ct.dicom_ingest import load_dicom_series_hu

        ct = load_dicom_series_hu(dicom_series)
        index = _brightest_index(ct.hu_zyx)
        position = _voxel_position_mm(index, ct.origin_xyz_mm, ct.direction, ct.spacing_zyx_mm)

        np.testing.assert_allclose(position, _fiducial_position_mm(), atol=1e-6)

    def test_right_side_edge_moves_to_the_left_end_of_axis_2(self, dicom_series: Path):
        from vasculature_digital_twin.ct.dicom_ingest import load_dicom_series_hu

        ct = load_dicom_series_hu(dicom_series)

        # Stored at the low-column (patient Right) edge; axis 2 now points Left.
        assert _darkest_columns(ct.hu_zyx[0]) == [COLUMNS - 2, COLUMNS - 1]

    def test_reorient_disabled_keeps_the_stored_layout(self, dicom_series: Path):
        from vasculature_digital_twin.ct.dicom_ingest import load_dicom_series_hu

        ct = load_dicom_series_hu(dicom_series, reorient=False)

        assert _darkest_columns(ct.hu_zyx[0]) == [0, 1]
        assert ct.anatomical_frame is None
